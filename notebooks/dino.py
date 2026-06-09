# 1. Clone your repository directly into the Kaggle working directory
!git clone https://github.com/peri-7/cv_project_saliency.git

import sys
import os
import torch

# 2. Append the cloned repository folder to Python's path
sys.path.append('/kaggle/working/cv_project_saliency/')

# 3. Verify hardware acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cloud Hardware active: {device}")

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from tqdm import tqdm

from src.dataset import LoraDataset
from src.models import DinoV2ViT
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.training_online import train_one_epoch_online, evaluate_model_online, test_model_online

# NOTE (Kaggle): `timm` is preinstalled, but downloading the SAM weights from the
# HuggingFace hub needs internet enabled (Settings -> Internet: on), or add the
# samvit_base_patch16.sa1b checkpoint as a Kaggle dataset and point timm at it.

print("---  Phase 1+2: Model Inference + Decoder Training (DinoV2 ViT-B) ---")

# Input normalization is held FIXED across all backbones for benchmark fairness.
image_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ground truth standardization
map_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])



