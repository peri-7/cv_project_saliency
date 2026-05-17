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
        super().train(False)