# DINOv3 ViT-B/16 (other-layers: blocks 7,8,10,11) — Decoder interpretability (weight analysis)
# Run from repo root:  python -m weight_analysis.dinov3_otherlayers_anal
# Or paste into a Kaggle notebook cell.

# 1. Clone your repository directly into the Kaggle working directory
get_ipython().system('git clone https://github.com/peri-7/cv_project_saliency.git')

import sys
import os
import torch

sys.path.append('/kaggle/working/cv_project_saliency/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cloud Hardware active: {device}")


import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import LoraDataset
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.models import DinoV3ViT
from weight_analysis.utils import run_analysis

# --- Edit these for your environment (Kaggle paths shown) --------------------
CKPT_PATH = '/kaggle/working/cv_project_saliency/saved_models/best_dinov3_decoder_otherlayers.pth'
DATA_ROOT = '/kaggle/input/datasets/roshan401/salicon'
OUT_DIR = '/kaggle/working'
# -----------------------------------------------------------------------------

MODEL_NAME = 'dinov3_otherlayers'
HIDDEN_DIM = 256  # ViT-B roster width
# This checkpoint was trained tapping blocks 7, 8, 10, 11 (0-indexed).
# All four taps share the /16 grid; variety is in depth, not resolution.
# DINOv3 uses patch16 so no crop is needed (480×640 is divisible by 16).
TAP_BLOCKS = [7, 8, 10, 11]
TAP_LABELS = ['block 7 (/16)', 'block 8 (/16)', 'block 10 (/16)', 'block 11 (/16)']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Input normalization held FIXED across all backbones (fairness rule).
image_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
map_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

val_dataset = LoraDataset(
    image_dir=os.path.join(DATA_ROOT, 'images/images/val'),
    maps_dir=os.path.join(DATA_ROOT, 'maps/val'),
    fixations_dir=os.path.join(DATA_ROOT, 'fixations/val'),
    image_transform=image_transform,
    map_transform=map_transform
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

extractor = DinoV3ViT(tap_blocks=TAP_BLOCKS).to(device)
decoder = Decoder(in_channels_list=extractor.out_channels,
                  hidden_dim=HIDDEN_DIM).to(device)
decoder.load_state_dict(torch.load(CKPT_PATH, map_location=device))
criterion = Composite_Loss().to(device)

run_analysis(extractor, decoder, val_loader, criterion, device,
             tap_labels=TAP_LABELS, model_name=MODEL_NAME, out_dir=OUT_DIR)
