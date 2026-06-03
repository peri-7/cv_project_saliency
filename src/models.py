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