import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    A lightweight, multi-scale convolutional decoding head utilizing 
    Group Normalization for stability under low-batch-memory constraints.
    """
    def __init__(self, in_channels_list, hidden_dim=256):
        super().__init__()
        
        total_in_channels = sum(in_channels_list)
        
        # 1. Dimensionality Reduction
        self.channel_compression = nn.Conv2d(
            in_channels=total_in_channels, 
            out_channels=hidden_dim, 
            kernel_size=1, 
            bias=False # Bias is redundant before Normalization
        )
        
        # GroupNorm splits the channels into groups (typically 32) to compute stats.
        # It is strictly superior to BatchNorm for batch sizes < 16.
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=hidden_dim)
        
        # 2. Spatial Smoothing
        self.spatial_conv = nn.Conv2d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=hidden_dim)
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