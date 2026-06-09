import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class ResNet(nn.Module):
    """
    A frozen ResNet-50 backbone designed explicitly for multi-scale feature extraction.
    This class bypasses the final classification head and taps into the intermediate
    convolutional blocks to provide low-level to high-level semantic features.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Load the pre-trained ResNet-50 model
        # Using the latest ImageNet V2 weights for the best baseline semantic features
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 2. Freeze all parameters strictly
        # This guarantees we do not destroy the pre-trained inductive biases and 
        # drastically reduces memory consumption during Phase 1.
        for param in base_model.parameters():
            param.requires_grad = False
            
        # Ensure the model remains in evaluation mode (affects BatchNorm/Dropout)
        base_model.eval()
        
        # 3. Define the multi-level extraction nodes
        # We sample uniformly across the network's depth to capture both 
        # bottom-up (contrast) and top-down (semantic) attention cues.
        return_nodes = {
            'layer1': 'feat_block1', # Outputs tensor of shape [B, 256, H/4, W/4]
            'layer2': 'feat_block2', # Outputs tensor of shape [B, 512, H/8, W/8]
            'layer3': 'feat_block3', # Outputs tensor of shape [B, 1024, H/16, W/16]
            'layer4': 'feat_block4'  # Outputs tensor of shape [B, 2048, H/32, W/32]
        }
        
        # 4. Create the automated PyTorch feature extractor
        self.extractor = create_feature_extractor(base_model, return_nodes=return_nodes)
        
        # A helper property so the Decoder knows exactly how many channels to expect
        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A batch of RGB images, shape [B, 3, H, W].
                              Must be normalized using ImageNet mean/std.
                              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        Returns:
            dict: A dictionary containing the 4 intermediate feature tensors.
        """
        # We wrap the forward pass in torch.no_grad() as a secondary failsafe 
        # to ensure absolutely no memory is wasted tracking gradients.
        with torch.no_grad():
            features = self.extractor(x)
            
        return features

    def train(self, mode=True):
        """
        Override the default train() method.
        This is a safety lock. Even if someone accidentally calls model.train()
        in the notebook, this ensures the backbone remains strictly in eval mode,
        preventing BatchNorm running statistics from being corrupted.
        """
        return super().train(False)


class SamViT(nn.Module):
    """
    A frozen SAM (Segment Anything Model) ViT image encoder, exposed through the
    exact same multi-scale contract as `ResNet` so the shared `Decoder` consumes
    it unchanged: frozen params, locked eval() mode, a dict of feature tensors,
    and an `out_channels` list.

    KEY DIFFERENCE FROM RESNET: SAM's encoder is a *plain* (non-hierarchical) ViT.
    Where ResNet gives four genuinely different resolutions (/4, /8, /16, /32),
    every SAM transformer block operates at the SAME /16 resolution. To still hand
    the decoder a comparable "low-level -> high-level" multi-scale stack, we tap
    four evenly-spaced block depths. The multi-scale variety therefore lives in
    abstraction *depth*, not spatial *resolution* (all four taps share one /16
    grid). The decoder handles this natively: it upsamples-and-concatenates to the
    finest map, and here the maps are already the same size.

    This 4-block depth-tap is the shared contract for the whole ViT-B comparison
    family (ViT / DINOv2 / MAE / CLIP / SAM), so every backbone gets the same
    multi-scale budget and only the pretraining objective varies.
    """

    def __init__(self,
                 model_name='samvit_base_patch16.sa1b',
                 tap_blocks=None):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "SamViT requires `timm` (pip install timm). It is preinstalled "
                "on Kaggle; install it locally for smoke tests."
            ) from e

        # 1. Load the SAM-pretrained ViT image encoder at its NATIVE size (1024).
        #    num_classes=0 drops the classification head. We deliberately do NOT
        #    pass img_size: building at native size lets the pretrained 64x64
        #    position embedding and the 127-length relative-position tables load
        #    cleanly, and timm resamples BOTH to the actual input grid at runtime
        #    (forward_features -> resample_abs_pos_embed_nhwc; attention ->
        #    get_rel_pos's F.interpolate). So non-square 480x640 -> 30x40 just works.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )

        # 2. Freeze every parameter (same rationale as the ResNet backbone:
        #    preserve the pretrained features; only the decoder ever trains).
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # 3. Choose four tap depths spread across the block stack (low -> high level).
        #    For ViT-B's 12 blocks this is blocks 2, 5, 8, 11 (0-indexed).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 4. Every tap is the plain-ViT embedding width (768 for ViT-B). The
        #    Decoder sums these, so out_channels mirrors the ResNet helper exactly.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        # 5. Register forward hooks that stash each tapped block's output so a single
        #    forward pass yields all four scales without re-running the backbone.
        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A batch of RGB images, shape [B, 3, H, W],
                              ImageNet-normalized (same normalization as ResNet,
                              held fixed across backbones for benchmark fairness).
        Returns:
            dict: four [B, embed_dim, H/16, W/16] feature tensors, ordered
                  shallow -> deep, matching the ResNet output contract.
        """
        self._features = {}
        # Failsafe no_grad (mirrors ResNet); forward_features drives patch embed
        # + pos embed + the block stack, and the hooks capture the four taps.
        with torch.no_grad():
            self.backbone.forward_features(x)

        # SAM ViT blocks keep spatial dims as [B, H, W, C]; the Decoder expects
        # channels-first, so permute each tapped feature to [B, C, H, W].
        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]
            features[f'sam_block{block_idx}'] = feat.permute(0, 3, 1, 2).contiguous()
        return features

    def train(self, mode=True):
        """
        Safety lock: keep the frozen backbone permanently in eval mode, even if
        someone calls model.train() in the notebook (see the ResNet override).
        """
        return super().train(False)


class MaeViT(nn.Module):
    """
    A frozen MAE (Masked Autoencoder) ViT-B image encoder, exposed through the
    exact same multi-scale contract as `ResNet` and `SamViT`: frozen params,
    locked eval() mode, a dict of feature tensors, and an `out_channels` list.

    MAE was pretrained via self-supervised masked autoencoding on ImageNet: 75%
    of patches are masked and the encoder learns to produce representations that
    let the decoder reconstruct the missing pixels. This forces the encoder to
    understand spatial structure and semantics without any labels.

    KEY DIFFERENCE FROM SAM: MAE uses a *standard* (vanilla) ViT with a CLS
    token. Each block outputs a token sequence [B, 1+N, C] where the first
    token is CLS and the remaining N tokens are the patch embeddings laid out
    in raster order. To produce the spatial [B, C, H, W] feature maps the
    decoder expects, we:
      1. Drop the CLS token  →  [B, N, C]
      2. Reshape to grid     →  [B, H_p, W_p, C]
      3. Permute to BCHW     →  [B, C, H_p, W_p]

    SAM's encoder, by contrast, uses windowed attention and keeps features as
    spatial [B, H, W, C] with no CLS token, so SamViT only needs a permute.

    This CLS-drop + reshape pattern will be shared by all future standard-ViT
    backbones (supervised ViT-B, CLIP ViT-B).
    """

    def __init__(self,
                 model_name='vit_base_patch16_224.mae',
                 tap_blocks=None):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "MaeViT requires `timm` (pip install timm). It is preinstalled "
                "on Kaggle; install it locally for smoke tests."
            ) from e

        # 1. Load the MAE-pretrained ViT-B encoder.
        #    num_classes=0 drops the classification head.
        #    Unlike SAM (native 1024×1024), MAE's native resolution is 224×224.
        #    By passing `dynamic_img_size=True`, timm will automatically interpolate
        #    the absolute position embeddings to match any input size (like 480×640)
        #    at runtime, preventing patch_embed shape assertion errors.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True
        )

        # 2. Freeze every parameter (same rationale as ResNet and SAM:
        #    preserve the pretrained features; only the decoder ever trains).
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # 3. Store the patch size for the token→grid reshape in forward().
        #    For ViT-B patch16 this is (16, 16).
        self.patch_size = base_model.patch_embed.patch_size

        # 4. Choose four tap depths spread across the block stack (low → high).
        #    For ViT-B's 12 blocks: blocks 2, 5, 8, 11 (0-indexed).
        #    This is the SAME tap contract as SamViT for structural fairness.
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 5. Every tap is the plain-ViT embedding width (768 for ViT-B).
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        # 6. Register forward hooks that stash each tapped block's output.
        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A batch of RGB images, shape [B, 3, H, W],
                              ImageNet-normalized (same normalization as all
                              other backbones, held fixed for benchmark fairness).
        Returns:
            dict: four [B, embed_dim, H/16, W/16] feature tensors, ordered
                  shallow → deep, matching the ResNet/SamViT output contract.
        """
        self._features = {}
        B, _, H, W = x.shape

        # Calculate the spatial grid dimensions from input size and patch size.
        H_p = H // self.patch_size[0]   # 480 // 16 = 30
        W_p = W // self.patch_size[1]   # 640 // 16 = 40

        # Failsafe no_grad (mirrors ResNet/SamViT).
        with torch.no_grad():
            self.backbone.forward_features(x)

        # Each hooked block output has shape [B, 1+N, C] where:
        #   1   = CLS token (global, non-spatial — we discard it)
        #   N   = H_p * W_p = number of patch tokens
        #   C   = embed_dim (768)
        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]         # [B, 1+N, C]
            feat = feat[:, 1:, :]               # Drop CLS → [B, N, C]
            feat = feat.reshape(B, H_p, W_p, -1)  # → [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # → [B, C, H_p, W_p]
            features[f'mae_block{block_idx}'] = feat
        return features

    def train(self, mode=True):
        """
        Safety lock: keep the frozen backbone permanently in eval mode, even if
        someone calls model.train() in the notebook (see the ResNet override).
        """
        return super().train(False)


class ClipViT(nn.Module):
    """
    A frozen CLIP ViT-B/16 image encoder (OpenAI's original weights), exposed
    through the exact same multi-scale contract as all other backbones.

    CLIP was pretrained via *contrastive image–text learning* on 400M
    image–caption pairs from the internet.  It learned to align visual and
    textual semantics, so its features are strongly biased toward objects and
    concepts that humans naturally describe in language — which may correlate
    with where humans look (saliency).

    Architecturally this is a standard (vanilla) ViT-B/16 — identical structure
    to MaeViT — so the forward path is the same: drop CLS token, reshape patch
    tokens to a spatial grid, permute to channels-first.  Only the pretrained
    weights differ.
    """

    def __init__(self,
                 model_name='vit_base_patch16_clip_224.openai',
                 tap_blocks=None):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "ClipViT requires `timm` (pip install timm). It is preinstalled "
                "on Kaggle; install it locally for smoke tests."
            ) from e

        # 1. Load the CLIP-pretrained ViT-B/16 image encoder.
        #    Native resolution is 224×224; dynamic_img_size=True lets timm
        #    interpolate position embeddings to our 480×640 input at runtime.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

        # 2. Freeze every parameter.
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # 3. Store patch size for token→grid reshape (16, 16).
        self.patch_size = base_model.patch_embed.patch_size

        # 4. Four evenly-spaced tap depths (blocks 2, 5, 8, 11 for ViT-B/12).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 5. out_channels: [768, 768, 768, 768] for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        # 6. Forward hooks to capture intermediate block outputs.
        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, 3, H, W], ImageNet-normalized.
        Returns:
            dict: four [B, 768, H/16, W/16] feature tensors.
        """
        self._features = {}
        B, _, H, W = x.shape
        H_p = H // self.patch_size[0]
        W_p = W // self.patch_size[1]

        with torch.no_grad():
            self.backbone.forward_features(x)

        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]                        # [B, 1+N, C]
            feat = feat[:, 1:, :]                              # Drop CLS
            feat = feat.reshape(B, H_p, W_p, -1)              # → [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()      # → [B, C, H_p, W_p]
            features[f'clip_block{block_idx}'] = feat
        return features

    def train(self, mode=True):
        return super().train(False)


class DinoV2ViT(nn.Module):
    """
    A frozen DINOv2 ViT-B/14 image encoder, exposed through the exact same
    multi-scale contract as all other backbones.

    DINOv2 was pretrained via self-distillation (no labels) on a curated 142M-image
    dataset (LVD-142M). Its patch tokens are known to carry strong local spatial
    structure, making them particularly well-suited for dense prediction tasks like
    saliency.

    KEY DIFFERENCE FROM THE OTHER ViT-B BACKBONES: DINOv2 uses patch size 14
    instead of 16. 480×640 is not divisible by 14 (480/14 ≈ 34.3, 640/14 ≈ 45.7),
    so the input is cropped to 476×630 (34×14, 45×14) before the patch embedding.
    The resulting feature grid is 34×45 instead of 30×40. The decoder handles
    arbitrary spatial sizes, so this is transparent downstream.

    Architecturally this is a standard ViT with a CLS token — identical forward
    path to MaeViT and ClipViT: drop CLS, reshape patch tokens to grid, permute
    to channels-first.
    """

    def __init__(self,
                 model_name='vit_base_patch14_dinov2.lvd142m',
                 tap_blocks=None):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "DinoV2ViT requires `timm` (pip install timm). It is preinstalled "
                "on Kaggle; install it locally for smoke tests."
            ) from e

        # 1. Load the DINOv2-pretrained ViT-B/14 encoder.
        #    Native resolution is 518×518; dynamic_img_size=True lets timm
        #    interpolate position embeddings to our cropped 476×630 grid at runtime.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

        # 2. Freeze every parameter.
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # 3. Store patch size (14, 14) for the crop and token→grid reshape.
        self.patch_size = base_model.patch_embed.patch_size

        # 4. Four evenly-spaced tap depths (blocks 2, 5, 8, 11 for ViT-B/12).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 5. out_channels: [768, 768, 768, 768] for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        # 6. Forward hooks to capture intermediate block outputs.
        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, 3, H, W], ImageNet-normalized.
        Returns:
            dict: four [B, 768, H_p, W_p] feature tensors where
                  H_p = (H // 14), W_p = (W // 14).
        """
        self._features = {}
        B, _, H, W = x.shape

        # Crop to the largest spatial size divisible by patch size 14.
        H_p = H // self.patch_size[0]   # 480 // 14 = 34
        W_p = W // self.patch_size[1]   # 640 // 14 = 45
        H_crop = H_p * self.patch_size[0]  # 476
        W_crop = W_p * self.patch_size[1]  # 630
        if H != H_crop or W != W_crop:
            x = x[:, :, :H_crop, :W_crop]

        with torch.no_grad():
            self.backbone.forward_features(x)

        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]                        # [B, 1+N, C]
            feat = feat[:, 1:, :]                              # Drop CLS → [B, N, C]
            feat = feat.reshape(B, H_p, W_p, -1)              # → [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()      # → [B, C, H_p, W_p]
            features[f'dino_block{block_idx}'] = feat
        return features

    def train(self, mode=True):
        return super().train(False)


class DinoV3ViT(nn.Module):
    """
    A frozen DINOv3 ViT-B/16 image encoder, exposed through the exact same
    multi-scale contract as all other backbones.

    DINOv3 (Meta, 2025) is the successor to DINOv2: self-supervised, trained on
    the much larger LVD-1689M corpus, with a Gram-anchoring objective that yields
    notably sharper, more consistent dense patch features — making it a strong
    candidate for dense prediction tasks like saliency.

    TWO DIFFERENCES FROM DinoV2ViT:
      1. PATCH SIZE 16 (not 14). 480×640 IS divisible by 16, so — unlike DINOv2 —
         no input crop is needed; the feature grid is a clean 30×40, matching the
         rest of the patch-16 ViT family.
      2. REGISTER TOKENS. DINOv3's token sequence is
         [CLS, reg_1, ..., reg_R, patch_1, ..., patch_N] — register tokens sit
         BETWEEN the CLS token and the patch tokens. So we cannot just drop a
         single leading token (as MaeViT/ClipViT do). Instead we take the LAST
         H_p*W_p tokens, which are always the spatial patch tokens regardless of
         how many prefix (CLS + register) tokens precede them.
    """

    def __init__(self,
                 model_name='vit_base_patch16_dinov3.lvd1689m',
                 tap_blocks=None):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "DinoV3ViT requires `timm` (pip install timm). It is preinstalled "
                "on Kaggle; install it locally for smoke tests."
            ) from e

        # 1. Load the DINOv3-pretrained ViT-B/16 encoder.
        #    Native resolution is 256×256; dynamic_img_size=True lets timm
        #    interpolate position embeddings to our 480×640 input at runtime.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

        # 2. Freeze every parameter.
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # 3. Store patch size (16, 16) for the token→grid reshape.
        self.patch_size = base_model.patch_embed.patch_size

        # 4. Four evenly-spaced tap depths (blocks 2, 5, 8, 11 for ViT-B/12).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 5. out_channels: [768, 768, 768, 768] for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        # 6. Forward hooks to capture intermediate block outputs.
        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, 3, H, W], ImageNet-normalized.
        Returns:
            dict: four [B, 768, H/16, W/16] feature tensors.
        """
        self._features = {}
        B, _, H, W = x.shape
        H_p = H // self.patch_size[0]   # 480 // 16 = 30
        W_p = W // self.patch_size[1]   # 640 // 16 = 40
        N = H_p * W_p

        with torch.no_grad():
            self.backbone.forward_features(x)

        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]                        # [B, prefix+N, C]
            feat = feat[:, -N:, :]                             # keep patch tokens → [B, N, C]
            feat = feat.reshape(B, H_p, W_p, -1)              # → [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()      # → [B, C, H_p, W_p]
            features[f'dinov3_block{block_idx}'] = feat
        return features

    def train(self, mode=True):
        return super().train(False)


class ViT(nn.Module):
    """
    A frozen ViT-B/16 backbone for multi-scale feature extraction.
    Follows the same contract as ResNet and SamViT:
    frozen params, locked eval(), dict of feature tensors, out_channels list.

    KEY DIFFERENCE FROM SAMVIT: Plain ViT is a token-sequence model [B, 1+N, C]
    with a CLS token. We drop the CLS token and reshape the remaining N patches
    into a spatial grid [B, C, H/16, W/16].
    """

    def __init__(self,
                 model_name='vit_base_patch16_224',
                 tap_blocks=None):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "ViT requires `timm` (pip install timm). It is preinstalled "
                "on Kaggle; install it locally for smoke tests."
            ) from e

        # 1. Load the pretrained ViT encoder. num_classes=0 drops the classification head.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=(480, 640)
        )

        # 2. Freeze all parameters and lock eval mode.
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # 3. Choose four tap depths spread across the block stack (low -> high level).
        #    For ViT-B's 12 blocks this is blocks 2, 5, 8, 11 (0-indexed).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 4. Every tap is the plain-ViT embedding width (768 for ViT-B).
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        # 5. Register forward hooks to capture each tapped block's output.
        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, 3, H, W], ImageNet-normalized.
        Returns:
            dict: four [B, embed_dim, H/16, W/16] feature tensors, shallow -> deep.
        """
        self._features = {}
        with torch.no_grad():
            self.backbone.forward_features(x)

        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]       # [B, 1+N, C] — includes CLS token
            feat = feat[:, 1:, :]             # drop CLS → [B, N, C]
            B, N, C = feat.shape
            H, W = x.shape[2] // 16, x.shape[3] // 16
            feat = feat.reshape(B, H, W, C)   # → [B, H/16, W/16, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # → [B, C, H/16, W/16]
            features[f'vit_block{block_idx}'] = feat

        return features

    def train(self, mode=True):
        """Safety lock: keep the frozen backbone permanently in eval mode."""
        return super().train(False)