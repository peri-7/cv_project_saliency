"""
Decoder interpretability for the ResNet-50 baseline (see weight_analysis/PLAN.md).

Run from the repo root as a module (same import contract as local_tests/):
    python -m weight_analysis.resnet_anal
"""

import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import LoraDataset
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.models import ResNet
from weight_analysis.utils import run_analysis

# --- Edit these for your environment (Kaggle paths shown) --------------------
CKPT_PATH = '/kaggle/input/best_resnet_decoder128.pth'
DATA_ROOT = '/kaggle/input/datasets/roshan401/salicon'
OUT_DIR = '/kaggle/working'
# -----------------------------------------------------------------------------

MODEL_NAME = 'resnet'
HIDDEN_DIM = 128  # the 128-width ResNet decoder beat the 256 one (PLAN.md)
TAP_LABELS = ['stage1 (/4)', 'stage2 (/8)', 'stage3 (/16)', 'stage4 (/32)']

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

extractor = ResNet().to(device)
decoder = Decoder(in_channels_list=extractor.out_channels,
                  hidden_dim=HIDDEN_DIM).to(device)
decoder.load_state_dict(torch.load(CKPT_PATH, map_location=device))
criterion = Composite_Loss().to(device)

run_analysis(extractor, decoder, val_loader, criterion, device,
             tap_labels=TAP_LABELS, model_name=MODEL_NAME, out_dir=OUT_DIR)
