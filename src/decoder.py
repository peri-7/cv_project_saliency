import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    A lightweight, multi-scale convolutional decoding head.
    It upsamples deep feature maps, concatenates them with shallow maps,
    and learns a spatial representation of visual attention.
    """
    def __init__(self, in_channels_list, hidden_dim=256):
        """
        Args:
            in_channels_list (list of int): The number of channels for each of the 
                                            extracted backbone layers.
                                            (e.g., [256, 512, 1024, 2048] for ResNet-50)
            hidden_dim (int): The number of channels to compress the concatenated features into.
        """
        super().__init__()
        
        # Calculate the total number of channels once all feature maps are concatenated
        total_in_channels = sum(in_channels_list)
        
        # 1. The Dimensionality Reduction & Feature Selection Layer
        # A 1x1 convolution acts as a learnable weighted sum across the channels.
        # This allows the network to effectively "zero out" useless backbone layers
        # and prioritize the most predictive semantic levels.
        self.channel_compression = nn.Conv2d(
            in_channels=total_in_channels, 
            out_channels=hidden_dim, 
            kernel_size=1, 
            bias=True
        )
        
        # 2. The Spatial Smoothing Layer
        # A standard 3x3 convolution to refine the spatial relationships of the fused features.
        # STRICT RULE: No nn.BatchNorm2d() is used here.
        self.spatial_conv = nn.Conv2d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1, 
            bias=True
        )
        
        self.activation = nn.ReLU(inplace=True)
        
        # 3. The Logit Projection Head
        # Projects the hidden dimension down to a single 2D spatial map.
        self.output_head = nn.Conv2d(
            in_channels=hidden_dim, 
            out_channels=1, 
            kernel_size=1, 
            bias=True
        )

    def forward(self, features):
        """
        Args:
            features (list or dict of torch.Tensor): The multi-scale tensors from the backbone.
        Returns:
            torch.Tensor: The raw, unnormalized 2D spatial logits (P_raw).
        """
        # If the extractor returned a dictionary, convert the values to a list
        if isinstance(features, dict):
            features = list(features.values())
            
        # The target spatial resolution is dictated by the first (shallowest) feature map
        target_size = features[0].shape[2:] # (Height, Width)
        
        upsampled_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                # Upsample deeper, lower-resolution maps using bilinear interpolation
                upsampled = F.interpolate(
                    feat, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                upsampled_features.append(upsampled)
            else:
                upsampled_features.append(feat)
                
        # Concatenate all maps along the channel dimension (dim=1)
        fused_tensor = torch.cat(upsampled_features, dim=1)
        
        # Pass through the convolutional blocks
        x = self.channel_compression(fused_tensor)
        x = self.activation(x)
        
        x = self.spatial_conv(x)
        x = self.activation(x)
        
        # Output the raw spatial logits (1 channel)
        logits = self.output_head(x)
        
        return logits