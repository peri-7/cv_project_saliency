import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels):
    """Largest #groups <= 32 that divides `channels` (GroupNorm constraint)."""
    return max(g for g in range(1, 33) if channels % g == 0)


class Decoder(nn.Module):
    """
    A lightweight, multi-scale convolutional decoding head utilizing 
    Group Normalization for stability under low-batch-memory constraints.
    """
    def __init__(self, in_channels_list, hidden_dim=512):
        super().__init__()
        
        total_in_channels = sum(in_channels_list)

        # GroupNorm needs hidden_dim divisible by num_groups. Prefer 32 groups,
        # but fall back to the largest divisor <= 32 so any hidden_dim (e.g. for a
        # different backbone) constructs without crashing. Always >= 1.
        num_groups = max(g for g in range(1, 33) if hidden_dim % g == 0)

        # 1. Dimensionality Reduction
        self.channel_compression = nn.Conv2d(
            in_channels=total_in_channels, 
            out_channels=hidden_dim, 
            kernel_size=1, 
            bias=False # Bias is redundant before Normalization
        )
        
        # GroupNorm splits the channels into groups (typically 32) to compute stats.
        # It is strictly superior to BatchNorm for batch sizes < 16.
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
        
        # 2. Spatial Smoothing
        self.spatial_conv = nn.Conv2d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        
        # 3. Logit Projection
        self.output_head = nn.Conv2d(
            in_channels=hidden_dim, 
            out_channels=1, 
            kernel_size=1, 
            bias=True
        )

    def forward(self, features):
        if isinstance(features, dict):
            features = list(features.values())
            
        target_size = features[0].shape[2:] 
        
        upsampled_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                upsampled = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                upsampled_features.append(upsampled)
            else:
                upsampled_features.append(feat)
                
        fused_tensor = torch.cat(upsampled_features, dim=1)
        
        # Forward pass with GroupNorm
        x = self.channel_compression(fused_tensor)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.spatial_conv(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        logits = self.output_head(x)

        return logits


class _ConvBlock(nn.Module):
    """Conv3x3 (bias=False) -> GroupNorm -> ReLU. The basic refinement unit
    of `ConvUpDecoder`."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=_num_groups(out_channels),
                                 num_channels=out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class ConvUpDecoder(nn.Module):
    """
    An upgraded convolutional decoding head for the LoRA experiment (see
    lora/LORA.md). NOT part of the frozen-backbone fairness study — the roster
    keeps the minimal `Decoder` above.

    The core upgrade over `Decoder` is distributing the upsampling across
    multiple LEARNED stages instead of one big bilinear stretch at the end:

      - `Decoder` outputs a logit at the feature resolution (/16 for the ViT
        family, 30x40) and the training loop bilinearly jumps 16x to 480x640
        with zero learned refinement.
      - `ConvUpDecoder` refines at /16, then progressively upsamples
        /16 -> /8 -> /4 with a learned conv block at each stage that can fix
        blurring artifacts before they propagate. The final bilinear jump in
        the training loop is then only 4x.

    Topology note: this is progressive upsampling (SETR-PUP-style), NOT FPN —
    all four ViT taps share the same /16 grid, so there is no resolution
    pyramid to ladder over.

    Drop-in compatible with `Decoder`: same constructor signature, same
    dict-or-list forward input, returns raw logits (here [B, 1, H/4, W/4];
    the train/eval loops already F.interpolate any decoder output to GT size).
    Uses resize-then-conv (bilinear + 3x3) rather than transposed convs to
    avoid checkerboard artifacts, and GroupNorm throughout (small-batch
    rationale, same as `Decoder`).
    """
    def __init__(self, in_channels_list, hidden_dim=256):
        super().__init__()

        total_in_channels = sum(in_channels_list)
        mid_dim = hidden_dim // 2
        low_dim = hidden_dim // 4

        # 1. Fuse the multi-scale taps: concat -> 1x1 conv -> hidden_dim.
        self.channel_compression = nn.Conv2d(total_in_channels, hidden_dim,
                                             kernel_size=1, bias=False)
        self.fuse_norm = nn.GroupNorm(num_groups=_num_groups(hidden_dim),
                                      num_channels=hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        # 2. Refine at the native feature resolution (/16).
        self.refine = _ConvBlock(hidden_dim, hidden_dim)

        # 3. Progressive upsampling with channel taper:
        #    /16 -> /8 (hidden -> hidden/2), /8 -> /4 (hidden/2 -> hidden/4).
        #    The bilinear resize happens in forward(); each conv block then
        #    sharpens the stretched features before the next stage.
        self.up_block1 = _ConvBlock(hidden_dim, mid_dim)
        self.up_block2 = _ConvBlock(mid_dim, low_dim)

        # 4. Logit head at /4.
        self.output_head = nn.Conv2d(low_dim, 1, kernel_size=1, bias=True)

    def forward(self, features):
        if isinstance(features, dict):
            features = list(features.values())

        target_size = features[0].shape[2:]

        upsampled_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear',
                                     align_corners=False)
            upsampled_features.append(feat)

        fused_tensor = torch.cat(upsampled_features, dim=1)

        x = self.channel_compression(fused_tensor)
        x = self.fuse_norm(x)
        x = self.activation(x)

        x = self.refine(x)

        # /16 -> /8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up_block1(x)

        # /8 -> /4
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up_block2(x)

        logits = self.output_head(x)

        return logits