# DINOv2 ViT-B Kaggle
#  Smoke Test
# Runs on 32 images for 1 epoch to verify everything works before full training.

!git clone https://github.com/peri-7/cv_project_saliency.git

import sys
import os
import torch

sys.path.append('/kaggle/working/cv_project_saliency/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cloud Hardware active: {device}")

import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from tqdm import tqdm

from src.dataset import LoraDataset
from src.models import DinoV2ViT
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.training_online import train_one_epoch_online, evaluate_model_online

print("--- KAGGLE SMOKE TEST: DinoV2 ViT-B ---")
print("32 εικόνες, 1 εποχή — μόνο για έλεγχο σφαλμάτων.")

image_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

map_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

base_input_path = '/kaggle/input/datasets/roshan401/salicon'

full_train_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/train"),
    maps_dir=os.path.join(base_input_path, "maps/train"),
    fixations_dir=os.path.join(base_input_path, "fixations/train"),
    image_transform=image_transform,
    map_transform=map_transform
)

full_val_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/val"),
    maps_dir=os.path.join(base_input_path, "maps/val"),
    fixations_dir=os.path.join(base_input_path, "fixations/val"),
    image_transform=image_transform,
    map_transform=map_transform
)

mini_train_dataset = Subset(full_train_dataset, range(32))
mini_val_dataset = Subset(full_val_dataset, range(32))

train_loader = DataLoader(mini_train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(mini_val_dataset, batch_size=16, shuffle=False, num_workers=2)

extractor = DinoV2ViT().to(device)

decoder = Decoder(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)
criterion = Composite_Loss().to(device)
optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

epochs = 1
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)

print("Ξεκινάει το mini training...")

train_loss, kld, cc = train_one_epoch_online(extractor, decoder, train_loader, optimizer, criterion, device)
val_loss = evaluate_model_online(extractor, decoder, val_loader, criterion, device)

print(f"Smoke Test Epoch 1 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("✅ TO SMOKE TEST ΟΛΟΚΛΗΡΩΘΗΚΕ ΜΕ ΕΠΙΤΥΧΙΑ! ✅")
print("Ο κώδικας του CLIP δεν έχει σφάλματα. Μπορείς να τρέξεις το κανονικό clip.py με σιγουριά!")
