import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class ResNet(nn.Module):
    """Frozen ResNet-50 backbone. Taps the four conv stages for multi-scale features."""
    def __init__(self):
        super().__init__()

        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze the backbone; only the decoder trains.
        for param in base_model.parameters():
            param.requires_grad = False

        base_model.eval()

        # Four evenly-spaced conv stages: shallow (contrast) to deep (semantics).
        return_nodes = {
            'layer1': 'feat_block1',  # [B, 256, H/4, W/4]
            'layer2': 'feat_block2',  # [B, 512, H/8, W/8]
            'layer3': 'feat_block3',  # [B, 1024, H/16, W/16]
            'layer4': 'feat_block4'   # [B, 2048, H/32, W/32]
        }

        self.extractor = create_feature_extractor(base_model, return_nodes=return_nodes)

        # Channel counts of the four taps, for the decoder.
        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x):
        # x: ImageNet-normalized RGB batch [B, 3, H, W]. Returns a dict of 4 feature tensors.
        with torch.no_grad():
            features = self.extractor(x)

        return features

    def train(self, mode=True):
        # Keep the backbone in eval mode even if train() is called, so BatchNorm
        # running stats stay frozen.
        return super().train(False)


class SamViT(nn.Module):
    """Frozen SAM ViT-B image encoder, same multi-scale contract as ResNet.

    SAM is a plain (non-hierarchical) ViT, so every block runs at /16 instead of
    ResNet's four resolutions. We tap four evenly-spaced block depths (2/5/8/11)
    so the multi-scale signal comes from abstraction depth rather than resolution
    -- the shared tap contract for the whole ViT-B family. SAM keeps features as
    [B, H, W, C] with no CLS token, so forward() only needs a permute.
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

        # Build at SAM's native 1024 size so the pretrained position embedding and
        # relative-position tables load cleanly; timm resamples both to the actual
        # 480x640 -> 30x40 grid at runtime. num_classes=0 drops the head.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )

        # Freeze the backbone; only the decoder trains.
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # Four evenly-spaced tap depths (blocks 2, 5, 8, 11 for ViT-B's 12).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 768 channels per tap for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        # Hooks stash each tapped block's output so one forward pass yields all taps.
        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        # x: ImageNet-normalized RGB [B, 3, H, W]. Returns four
        # [B, embed_dim, H/16, W/16] tensors, shallow to deep.
        self._features = {}
        with torch.no_grad():
            self.backbone.forward_features(x)

        # SAM keeps features as [B, H, W, C]; permute to channels-first.
        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]
            features[f'sam_block{block_idx}'] = feat.permute(0, 3, 1, 2).contiguous()
        return features

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode even if train() is called.
        return super().train(False)


class MaeViT(nn.Module):
    """Frozen MAE ViT-B image encoder, same multi-scale contract as SamViT.

    MAE is a standard ViT with a CLS token: each block emits a token sequence
    [B, 1+N, C]. To get spatial [B, C, H, W] maps we drop the CLS token, reshape
    the N patch tokens to a grid, and permute to channels-first -- the same
    pattern reused by ClipViT and the supervised ViT.
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

        # MAE's native resolution is 224; dynamic_img_size=True interpolates the
        # position embeddings to our 480x640 input at runtime. num_classes=0 drops
        # the head.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True
        )

        # Freeze the backbone; only the decoder trains.
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # (16, 16) for ViT-B/16, used for the token->grid reshape.
        self.patch_size = base_model.patch_embed.patch_size

        # Four evenly-spaced tap depths (blocks 2, 5, 8, 11), same as SamViT.
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 768 channels per tap for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        # Hooks stash each tapped block's output.
        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        # x: ImageNet-normalized RGB [B, 3, H, W]. Returns four
        # [B, embed_dim, H/16, W/16] tensors, shallow to deep.
        self._features = {}
        B, _, H, W = x.shape
        H_p = H // self.patch_size[0]   # 480 // 16 = 30
        W_p = W // self.patch_size[1]   # 640 // 16 = 40

        with torch.no_grad():
            self.backbone.forward_features(x)

        # Block outputs are [B, 1+N, C]: drop the CLS token, reshape to a grid.
        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]                   # [B, 1+N, C]
            feat = feat[:, 1:, :]                         # drop CLS -> [B, N, C]
            feat = feat.reshape(B, H_p, W_p, -1)          # -> [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H_p, W_p]
            features[f'mae_block{block_idx}'] = feat
        return features

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode even if train() is called.
        return super().train(False)


class ClipViT(nn.Module):
    """Frozen CLIP ViT-B/16 image encoder (OpenAI weights).

    Architecturally identical to MaeViT (standard ViT with a CLS token); only the
    pretraining differs -- CLIP learned image-text alignment on 400M web pairs, so
    its features lean toward language-nameable objects. Same forward path: drop
    CLS, reshape patch tokens to a grid, permute to channels-first.
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

        # Native 224; dynamic_img_size=True interpolates position embeddings to 480x640.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        self.patch_size = base_model.patch_embed.patch_size  # (16, 16)

        # Four evenly-spaced tap depths (blocks 2, 5, 8, 11).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 768 channels per tap for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        # x: ImageNet-normalized RGB [B, 3, H, W]. Returns four
        # [B, 768, H/16, W/16] tensors.
        self._features = {}
        B, _, H, W = x.shape
        H_p = H // self.patch_size[0]
        W_p = W // self.patch_size[1]

        with torch.no_grad():
            self.backbone.forward_features(x)

        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]                   # [B, 1+N, C]
            feat = feat[:, 1:, :]                         # drop CLS
            feat = feat.reshape(B, H_p, W_p, -1)          # -> [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H_p, W_p]
            features[f'clip_block{block_idx}'] = feat
        return features

    def train(self, mode=True):
        return super().train(False)


class DinoV2ViT(nn.Module):
    """Frozen DINOv2 ViT-B/14 image encoder (self-distilled on LVD-142M).

    Same standard-ViT forward path as MaeViT/ClipViT, but patch size 14: 480x640
    isn't divisible by 14, so the input is cropped to 476x630 before patch
    embedding, giving a 34x45 grid instead of 30x40. The decoder accepts arbitrary
    spatial sizes, so this is transparent downstream.
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

        # Native 518; dynamic_img_size=True interpolates position embeddings to 476x630.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        self.patch_size = base_model.patch_embed.patch_size  # (14, 14)

        # Four evenly-spaced tap depths (blocks 2, 5, 8, 11).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 768 channels per tap for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        # x: ImageNet-normalized RGB [B, 3, H, W]. Returns four
        # [B, 768, H_p, W_p] tensors, with H_p = H // 14, W_p = W // 14.
        self._features = {}
        B, _, H, W = x.shape

        # Crop to the largest size divisible by patch size 14 (476x630).
        H_p = H // self.patch_size[0]      # 34
        W_p = W // self.patch_size[1]      # 45
        H_crop = H_p * self.patch_size[0]  # 476
        W_crop = W_p * self.patch_size[1]  # 630
        if H != H_crop or W != W_crop:
            x = x[:, :, :H_crop, :W_crop]

        with torch.no_grad():
            self.backbone.forward_features(x)

        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]                   # [B, 1+N, C]
            feat = feat[:, 1:, :]                         # drop CLS
            feat = feat.reshape(B, H_p, W_p, -1)          # -> [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H_p, W_p]
            features[f'dino_block{block_idx}'] = feat
        return features

    def train(self, mode=True):
        return super().train(False)


class DinoV3ViT(nn.Module):
    """Frozen DINOv3 ViT-B/16 image encoder (Meta 2025, self-supervised on LVD-1689M).

    Two differences from DinoV2ViT:
      1. Patch size 16 -- 480x640 divides cleanly, so no crop; grid is 30x40.
      2. Register tokens. The sequence is [CLS, reg_1..reg_R, patch_1..patch_N],
         so we can't just drop one leading token. We keep the last H_p*W_p tokens,
         which are always the patch tokens regardless of how many prefix tokens exist.
    """

    def __init__(self,
                 model_name='vit_base_patch16_dinov3',
                 tap_blocks=None):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "DinoV3ViT requires `timm` (pip install timm). It is preinstalled "
                "on Kaggle; install it locally for smoke tests."
            ) from e

        # Native 256; dynamic_img_size=True interpolates position embeddings to 480x640.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        self.patch_size = base_model.patch_embed.patch_size  # (16, 16)

        # Four evenly-spaced tap depths (blocks 2, 5, 8, 11).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 768 channels per tap for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        # x: ImageNet-normalized RGB [B, 3, H, W]. Returns four
        # [B, 768, H/16, W/16] tensors.
        self._features = {}
        B, _, H, W = x.shape
        H_p = H // self.patch_size[0]   # 30
        W_p = W // self.patch_size[1]   # 40
        N = H_p * W_p

        with torch.no_grad():
            self.backbone.forward_features(x)

        # Register tokens sit between CLS and patches, so keep the last N tokens.
        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]                   # [B, prefix+N, C]
            feat = feat[:, -N:, :]                        # keep patch tokens
            feat = feat.reshape(B, H_p, W_p, -1)          # -> [B, H_p, W_p, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H_p, W_p]
            features[f'dinov3_block{block_idx}'] = feat
        return features

    def train(self, mode=True):
        return super().train(False)


class ViT(nn.Module):
    """Frozen supervised ViT-B/16 backbone, same contract as ResNet and SamViT.

    Standard token-sequence ViT with a CLS token: drop CLS, reshape the N patch
    tokens to a spatial grid.
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

        # num_classes=0 drops the head; img_size fixes the position embedding grid.
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=(480, 640)
        )

        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.backbone = base_model

        # Four evenly-spaced tap depths (blocks 2, 5, 8, 11).
        num_blocks = len(base_model.blocks)
        if tap_blocks is None:
            tap_blocks = [
                num_blocks // 4 - 1,
                num_blocks // 2 - 1,
                (3 * num_blocks) // 4 - 1,
                num_blocks - 1,
            ]
        self.tap_blocks = tap_blocks

        # 768 channels per tap for ViT-B.
        embed_dim = base_model.embed_dim
        self.out_channels = [embed_dim] * len(tap_blocks)

        self._features = {}
        for slot, block_idx in enumerate(tap_blocks):
            base_model.blocks[block_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            self._features[slot] = output
        return hook

    def forward(self, x):
        # x: ImageNet-normalized RGB [B, 3, H, W]. Returns four
        # [B, embed_dim, H/16, W/16] tensors, shallow to deep.
        self._features = {}
        with torch.no_grad():
            self.backbone.forward_features(x)

        features = {}
        for slot, block_idx in enumerate(self.tap_blocks):
            feat = self._features[slot]                   # [B, 1+N, C]
            feat = feat[:, 1:, :]                         # drop CLS
            B, N, C = feat.shape
            H, W = x.shape[2] // 16, x.shape[3] // 16
            feat = feat.reshape(B, H, W, C)               # -> [B, H/16, W/16, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H/16, W/16]
            features[f'vit_block{block_idx}'] = feat

        return features

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode even if train() is called.
        return super().train(False)