"""
LoRA pathway for the DINOv3 + upgraded-decoder experiment (see lora/LORA.md).

Separate from training.py / training_online.py: those cover the frozen-backbone
roster, while LoRA backprops through the backbone trunk. Only train_one_epoch_lora
lives here; validation and final benchmarking reuse evaluate_model_online /
test_model_online from src/training_online.py unchanged (they run under
torch.no_grad(), which is correct at inference for LoRA too).

Why this avoids the forward hooks the frozen backbones in src/models.py use: with
gradient checkpointing enabled, each block's forward runs in no-grad mode
(activations are recomputed during backward), so a hook captures a tensor that is
detached from the autograd graph -- feeding those to the decoder would silently
train with zero LoRA gradients. Instead we use timm's forward_intermediates() API,
which returns the checkpoint outputs (those keep their grad_fn). train_one_epoch_lora
also checks on the first batch that LoRA gradients are non-zero and raises if not.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training_online import test_model_online


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a trainable low-rank adapter:

        forward(x) = base(x) + (alpha / r) * lora_B(lora_A(dropout(x)))

    The base layer is frozen. `lora_A` is kaiming-initialized and `lora_B` is
    ZERO-initialized, so the adapter starts as an exact no-op and the wrapped
    model's initial behaviour is identical to the pretrained one.
    """

    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad = False

        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Linear(base_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


class LoraDinoV3ViT(nn.Module):
    """
    A DINOv3 ViT-B/16 backbone with LoRA adapters injected into the attention
    layers. Same construction and token handling as the frozen `DinoV3ViT` in
    `src/models.py` (patch-16, register tokens -> keep the LAST H_p*W_p tokens,
    reshape, permute, out_channels=[768]*len(tap_blocks)), with three deliberate
    differences:

      1. The trunk runs WITH grad — no torch.no_grad() wrapper — so gradients
         reach the LoRA adapters. (Inference loops that wrap the call in
         no_grad still work; the trunk simply runs grad-free there.)
      2. train() is NOT overridden to force eval. ViT has only LayerNorm (no
         BatchNorm running stats to corrupt) and train mode is needed for LoRA
         dropout to be active.
      3. Optional gradient checkpointing on the trunk (`grad_checkpointing=
         True`) — the lever for fitting LoRA backprop on a 16 GB Kaggle GPU.

    Two independent parametric axes (orthogonal — you can LoRA all 12 blocks
    but tap only one, etc.):

      - `lora_blocks`: which block indices get LoRA adapters. Default: ALL
        blocks. The tapped blocks are where features are *read*, but block
        11's output depends on every block before it; adapting all 12 gives
        the trunk full freedom to reshape representations, at trivial extra
        parameter cost.
      - `tap_blocks`: which block indices features are *extracted* from.
        Default: the roster's evenly-spaced four (2/5/8/11 for ViT-B). Drives
        `out_channels`, which the decoder constructor consumes generically.
    """

    def __init__(self,
                 model_name='vit_base_patch16_dinov3',
                 tap_blocks=None,
                 lora_blocks=None,
                 lora_targets=('qkv', 'proj'),
                 r=8,
                 alpha=16,
                 lora_dropout=0.0,
                 grad_checkpointing=False):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "LoraDinoV3ViT requires `timm` (pip install timm). It is "
                "preinstalled on Kaggle; install it locally for smoke tests."
            ) from e

        # Load the DINOv3-pretrained ViT-B/16 encoder (gated weights — needs
        #    a prior huggingface_hub.login(); see lora/LORA.md).
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

        # Freeze EVERY base parameter first; only the LoRA adapters injected
        #    below (plus the decoder, externally) are ever trainable.
        for param in base_model.parameters():
            param.requires_grad = False

        self.patch_size = base_model.patch_embed.patch_size
        num_blocks = len(base_model.blocks)

        # Tap blocks: where features are read (same default contract as the
        #    frozen roster: blocks 2, 5, 8, 11 for ViT-B's 12).
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        # forward_intermediates returns features in ascending block order, so
        # keep tap_blocks sorted to zip them back to their indices correctly.
        self.tap_blocks = sorted(tap_blocks)

        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(self.tap_blocks)

        # LoRA blocks: where adapters are injected. Default: all blocks.
        if lora_blocks is None:
            lora_blocks = list(range(num_blocks))
        self.lora_blocks = sorted(lora_blocks)

        for block_idx in self.lora_blocks:
            attn = base_model.blocks[block_idx].attn
            for target in lora_targets:
                for attr in self._resolve_target(attn, target):
                    setattr(attn, attr, LoRALinear(
                        getattr(attn, attr), r=r, alpha=alpha, dropout=lora_dropout
                    ))

        # Gradient checkpointing: trade recompute for activation memory.
        self.grad_checkpointing = grad_checkpointing
        if grad_checkpointing:
            base_model.set_grad_checkpointing(True)
            # Reentrant checkpointing computes no gradients at all when none of
            # a segment's *inputs* require grad — and the patch embedding is
            # frozen, so block 0's input does not. The standard fix (same trick
            # as HF peft's enable_input_require_grads) is to flip requires_grad
            # on the embedding output so gradients flow into every checkpointed
            # segment regardless of the checkpoint implementation.
            base_model.patch_embed.register_forward_hook(
                lambda module, inputs, output: output.requires_grad_(True)
            )

        self.backbone = base_model

    @staticmethod
    def _resolve_target(attn, target):
        """Map a logical target name to the actual nn.Linear attribute(s) on
        the attention module — timm attention blocks use either a fused `qkv`
        or separate `q_proj`/`k_proj`/`v_proj` projections depending on the
        model family."""
        if target == 'qkv':
            if isinstance(getattr(attn, 'qkv', None), nn.Linear):
                return ['qkv']
            if isinstance(getattr(attn, 'q_proj', None), nn.Linear):
                return ['q_proj', 'k_proj', 'v_proj']
            raise AttributeError(
                f"Attention module {type(attn).__name__} has neither a fused "
                f"`qkv` nor separate `q_proj`/`k_proj`/`v_proj` Linear layers."
            )
        if target == 'proj':
            if isinstance(getattr(attn, 'proj', None), nn.Linear):
                return ['proj']
            raise AttributeError(
                f"Attention module {type(attn).__name__} has no `proj` Linear."
            )
        raise ValueError(f"Unknown LoRA target {target!r}; use 'qkv' or 'proj'.")

    def trainable_parameters(self):
        """The LoRA parameters — what the optimizer should cover on the
        backbone side (the base trunk is frozen)."""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        """Returns len(tap_blocks) tensors [B, 768, H/16, W/16], shallow to deep,
        matching the frozen roster's contract. x is ImageNet-normalized [B, 3, H, W].

        No torch.no_grad() here — that would kill the LoRA gradients.
        forward_intermediates collects each tapped block's output after the
        (possibly checkpointed) block call, so the tensors stay attached to the
        autograd graph.
        """
        B, _, H, W = x.shape
        H_p = H // self.patch_size[0]   # 480 // 16 = 30
        W_p = W // self.patch_size[1]   # 640 // 16 = 40
        N = H_p * W_p

        intermediates = self.backbone.forward_intermediates(
            x,
            indices=self.tap_blocks,
            output_fmt='NLC',
            intermediates_only=True,
            norm=False,
        )

        features = {}
        for block_idx, feat in zip(self.tap_blocks, intermediates):
            feat = feat[:, -N:, :]                        # keep patch tokens (skip CLS + registers)
            feat = feat.reshape(B, H_p, W_p, -1)          # -> [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H_p, W_p]
            features[f'dinov3_block{block_idx}'] = feat
        return features


def train_one_epoch_lora(extractor, decoder, dataloader, optimizer, criterion, device):
    """
    Like `train_one_epoch_online` in `src/training_online.py`, BUT the backbone
    forward runs WITH grad so the LoRA adapters receive gradients. The
    optimizer must cover both the decoder params and the LoRA params (see
    `LoraDinoV3ViT.trainable_parameters`). Returns the same
    (avg_loss, avg_kld, avg_cc) 3-tuple.

    Validation / final benchmarking should reuse `evaluate_model_online` /
    `test_model_online` unchanged — torch.no_grad() is correct at inference
    for LoRA too.
    """
    extractor.train()  # LoRA dropout active; ViT has no BatchNorm to corrupt
    decoder.train()

    total_loss_accum, kld_accum, cc_accum = 0.0, 0.0, 0.0
    first_batch = True

    for images, target_map, _ in dataloader:

        images = images.to(device)
        target_map = target_map.to(device)

        optimizer.zero_grad()

        # Extract with gradients enabled (the LoRA path needs them).
        features_dict = extractor(images)
        features = list(features_dict.values())

        raw_logits = decoder(features)

        matched_logits = F.interpolate(
            raw_logits,
            size=(target_map.shape[2], target_map.shape[3]),
            mode='bilinear', align_corners=False
        )

        loss, loss_kld, loss_cc, _ = criterion(matched_logits, target_map)

        loss.backward()

        # First-batch check: if the features get detached from the graph (e.g. a
        # hook/checkpointing interaction), LoRA would train silently at zero
        # gradient, so fail loudly instead. lora_B is zero-init, so on the first
        # step lora_A grads are zero but lora_B grads must be non-zero.
        if first_batch:
            lora_params = [p for n, p in extractor.named_parameters() if 'lora_' in n]
            if lora_params and not any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in lora_params
            ):
                raise RuntimeError(
                    "All LoRA gradients are zero/None after the first backward "
                    "pass — the backbone features are detached from the autograd "
                    "graph. Check that the extractor forward is NOT wrapped in "
                    "torch.no_grad() and that features are taken from checkpoint "
                    "outputs, not from hooks inside checkpointed blocks."
                )
            first_batch = False

        optimizer.step()

        total_loss_accum += loss.item()
        kld_accum += loss_kld.item()
        cc_accum += loss_cc.item()

    L = len(dataloader)
    avg_loss = total_loss_accum / L
    avg_kld = kld_accum / L
    avg_cc = cc_accum / L

    return avg_loss, avg_kld, avg_cc


def make_gaussian_center_bias(height, width, sigma_scale=0.25):
    """
    Builds a Gaussian center-bias probability map [1, 1, H, W] that sums to 1,
    for use as the Information Gain baseline. Humans fixate the image center
    far more than the periphery, so a center Gaussian is a much stronger (and
    more standard) IG reference than the uniform map — IG scores against it
    are correspondingly lower and more honest.

    The Gaussian is anisotropic: sigma is `sigma_scale` of each dimension
    (sigma_y = sigma_scale * H, sigma_x = sigma_scale * W), so the bias has
    the same relative spread regardless of aspect ratio.
    """
    ys = torch.arange(height, dtype=torch.float32) - (height - 1) / 2.0
    xs = torch.arange(width, dtype=torch.float32) - (width - 1) / 2.0
    sigma_y = sigma_scale * height
    sigma_x = sigma_scale * width

    gauss = torch.exp(
        -(ys.view(-1, 1) ** 2) / (2 * sigma_y ** 2)
        - (xs.view(1, -1) ** 2) / (2 * sigma_x ** 2)
    )
    gauss = gauss / gauss.sum()
    return gauss.view(1, 1, height, width)


@torch.no_grad()
def test_model_online_lora(extractor, decoder, dataloader, criterion, device,
                           sigma_scale=0.25):
    """
    Same as `test_model_online` (same 7-tuple `(loss, kld, cc, sim, nss, auc,
    ig)`), but Information Gain is computed against a GAUSSIAN center-bias
    baseline instead of the uniform map. Everything else is delegated to
    `test_model_online` unchanged, so the other six numbers are identical to
    what the roster benchmark would report.

    NOTE: the IG number is NOT comparable with roster runs benchmarked via
    plain `test_model_online` (uniform baseline) — center bias is a stronger
    reference, so IG comes out lower by construction.
    """
    # The GT resolution is immutable per project convention; read it off one
    # sample (dataset items are (image, map, fixation)) to size the baseline.
    sample_map = dataloader.dataset[0][1]
    height, width = sample_map.shape[-2:]
    baseline_prob = make_gaussian_center_bias(height, width, sigma_scale)

    return test_model_online(extractor, decoder, dataloader, criterion, device,
                             baseline_prob=baseline_prob)
