import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your custom modules strictly from the src/ layer
from src.dataset import RawDataset, FeatureDataset
from src.models import ResNet
from src.decoder import Decoder
from src.losses import UNETRSal_Loss
from src.training import train_one_epoch

# =====================================================================
# 0. SETUP: Create Mini Dummy Dataset for Immediate Testing
# =====================================================================
def create_dummy_mini_dataset():
    """Generates 5 synthetic images and maps to test tensor flow."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/maps", exist_ok=True)
    os.makedirs("data/fixations", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)
    
    for i in range(5):
        # 1. RGB Image (480x640)
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img.save(f"data/raw/COCO_val2014_00000000000{i}.jpg")
        
        # 2. Continuous Saliency Map (Grayscale)
        smap = Image.fromarray(np.random.randint(0, 255, (480, 640), dtype=np.uint8))
        smap.save(f"data/maps/COCO_val2014_00000000000{i}.png")
        
        # 3. Discrete Fixation Map (Binary)
        fmap = Image.fromarray(np.random.choice([0, 255], (480, 640)).astype(np.uint8))
        fmap.save(f"data/fixations/COCO_val2014_00000000000{i}.png")

    print("Dummy Mini-Dataset generated successfully.")

# =====================================================================
# 1. PHASE 1 TEST: Feature Extraction (The CPU/GPU Bottleneck)
# =====================================================================
def test_phase_1_extraction(device):
    print("\n--- Starting Phase 1: Feature Extraction ---")
    
    image_transform = transforms.Compose([
        transforms.Resize((480, 640)), # Ensure standard COCO size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_dataset = RawDataset(image_dir="data/raw", transform=image_transform)
    raw_loader = DataLoader(raw_dataset, batch_size=2, shuffle=False)
    
    extractor = ResNet().to(device)
    
    for batch_idx, (images, filenames) in enumerate(raw_loader):
        images = images.to(device)
        features_dict = extractor(images)
        
        # Save features individually to disk
        for i in range(images.size(0)):
            base_name = os.path.splitext(filenames[i])[0]
            save_path = f"data/features/{base_name}.pt"
            
            # Extract the specific item from the batch for each feature map
            single_feature_dict = {
                k: v[i:i+1].cpu() for k, v in features_dict.items()
            }
            torch.save(single_feature_dict, save_path)
            
    print(f"Extraction complete. Features saved to data/features/")
    
    # Return the out_channels so the decoder knows how to size itself
    return extractor.out_channels

# =====================================================================
# 2. PHASE 2 TEST: Decoder Optimization 
# =====================================================================
def test_phase_2_training(device, backbone_channels):
    print("\n--- Starting Phase 2: Decoder Training ---")
    
    # We resize ground truths to 480x640 to standardize the batch
    map_transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor()
    ])
    
    feature_dataset = FeatureDataset(
        features_dir="data/features",
        maps_dir="data/maps",
        fixations_dir="data/fixations",
        map_transform=map_transform
    )
    
    # Use a small batch size for testing
    feature_loader = DataLoader(feature_dataset, batch_size=2, shuffle=True)
    
    # Initialize the architecture
    decoder = Decoder(in_channels_list=backbone_channels, hidden_dim=128).to(device)
    criterion = UNETRSal_Loss().to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Run a single epoch to ensure gradients flow and loss converges
    print("Executing Epoch 1...")
    avg_loss, avg_kld, avg_cc = train_one_epoch(
        decoder, feature_loader, optimizer, criterion, device
    )
    
    print(f"Epoch 1 Complete!")
    print(f"Total Composite Loss: {avg_loss:.4f}")
    print(f"KLD Component: {avg_kld:.4f}")
    print(f"CC Component: {avg_cc:.4f}")
    print("\nPIPELINE TEST PASSED: Tensors matched perfectly and gradients computed.")

# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on hardware: {device}")
    
    create_dummy_mini_dataset()
    channels = test_phase_1_extraction(device)
    test_phase_2_training(device, channels)