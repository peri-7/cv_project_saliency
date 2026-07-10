import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import your custom modules strictly from the src/ layer
from src.dataset import RawDataset, FeatureDataset
from src.models import ResNet
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.training import train_one_epoch, evaluate_model, test_model


# 1. PHASE 1 TEST: Feature Extraction (The CPU/GPU Bottleneck)
def test_phase_1_extraction(device):
    print("\n--- Starting Phase 1: Feature Extraction ---")
    
    image_transform = transforms.Compose([
        transforms.Resize((480, 640)), # Ensure standard COCO size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    extractor = ResNet().to(device)
    with torch.no_grad():
        for x in ['train', 'test', 'val']:
            
            os.makedirs(f"mini_data/features/{x}", exist_ok=True)
            
            raw_dataset = RawDataset(image_dir=f"mini_data/images/{x}", transform=image_transform)
            raw_loader = DataLoader(raw_dataset, batch_size=2, shuffle=False)
            
            
            for batch_idx, (images, filenames) in enumerate(raw_loader):
                images = images.to(device)
                features_dict = extractor(images)
                
                # Save features individually to disk
                for i in range(images.size(0)):
                    base_name = os.path.splitext(filenames[i])[0]
                    save_path = f"mini_data/features/{x}/{base_name}.pt"
                    
                    # Extract the specific item from the batch for each feature map
                    single_feature_dict = {
                        k: v[i].cpu() for k, v in features_dict.items()
                    }
                    torch.save(single_feature_dict, save_path)
                    
    print(f"Extraction complete. Features saved to mini_data/features/")
    
    # Return the out_channels so the decoder knows how to size itself
    return extractor.out_channels


# 2. PHASE 2 TEST: Decoder Optimization 
def test_phase_2_training(device, backbone_channels):
    print("\n--- Starting Phase 2: Decoder Training ---")
    
    # We resize ground truths to 480x640 to standardize the batch
    map_transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor()
    ])
    
    train_dataset = FeatureDataset(
        features_dir="mini_data/features/train",
        maps_dir="mini_data/maps/train",
        fixations_dir="mini_data/fixations/train",
        map_transform=map_transform
    )
    # Use a small batch size for testing
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    val_dataset = FeatureDataset(
        features_dir="mini_data/features/val",
        maps_dir="mini_data/maps/val",
        fixations_dir="mini_data/fixations/val",
        map_transform=map_transform
    )
    
    # Use a small batch size for testing
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Initialize the architecture
    decoder = Decoder(in_channels_list=backbone_channels, hidden_dim=128).to(device)
    criterion = Composite_Loss().to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-4)
    patience = 0
    val_min = float('inf')
    
    # Run a single epoch to ensure gradients flow and loss converges
    print("Starting training 10 epochs...")
    epoch_iterator = tqdm(range(4), desc='Training Epochs')
    for epoch in epoch_iterator:   
        train_loss, kld, cc = train_one_epoch(decoder, train_loader, optimizer, criterion, device)
        
        val_loss = evaluate_model(decoder, val_loader, criterion, device)
        epoch_iterator.set_postfix({'Train Loss': f"{train_loss:.4f}", 'Val Loss': f"{val_loss:.4f}"})
        
        if val_loss < val_min:
            patience = 0
            val_min = val_loss
        else:
            patience += 1
            
        if patience > 3: 
            print(f'Early Stopping on {epoch}')
            break   
        
    print('-'*50)
    print("Training ended succesfully")
    
    # we use the validation set because test set misses the maps
    test_dataset = FeatureDataset(
        features_dir="mini_data/features/val",
        maps_dir="mini_data/maps/val",
        fixations_dir="mini_data/fixations/val",
        map_transform=map_transform
    )
    # Use a small batch size for testing
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig = test_model(decoder, test_loader, criterion, device)
    print(f'''
    Test Loss: {avg_loss}, KLD: {avg_kld}, CC: {avg_cc}, SIM: {avg_sim}
    NSS: {avg_nss}, AUC: {avg_auc}, IG: {avg_ig}
    ''')
    

# EXECUTION
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on hardware: {device}")
    
    channels = test_phase_1_extraction(device)
    test_phase_2_training(device, channels)