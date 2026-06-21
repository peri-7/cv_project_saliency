# Evaluation-only script for the 2-tap LoRA run.
# Loads the best checkpoint saved during training and computes the final
# metrics — use this when the training session timed out before evaluation.
#
# Run on Kaggle:
#   !rm -rf /kaggle/working/cv_project_saliency
#   !git clone https://ghp_...@github.com/peri-7/cv_project_saliency.git
#   !HF_TOKEN="hf_..." python /kaggle/working/cv_project_saliency/lora/dino3_lora_2tap_eval.py

# --- Point this at the best checkpoint from the training run ---
CKPT_PATH = '/kaggle/working/cv_project_saliency/lora/best_dinov3_lora_upgraded_2tap.pth'

TAP_BLOCKS = [10, 11]

import sys
import os
import torch

sys.path.append('/kaggle/working/cv_project_saliency/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cloud Hardware active: {device}")

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import LoraDataset
from src.decoder import ConvUpDecoder
from src.losses import Composite_Loss
from src.lora import LoraDinoV3ViT, test_model_online_lora

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

val_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/val"),
    maps_dir=os.path.join(base_input_path, "maps/val"),
    fixations_dir=os.path.join(base_input_path, "fixations/val"),
    image_transform=image_transform,
    map_transform=map_transform
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# Rebuild the same architecture used during training
extractor = LoraDinoV3ViT(tap_blocks=TAP_BLOCKS).to(device)
decoder = ConvUpDecoder(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)
criterion = Composite_Loss().to(device)

# Load the best checkpoint
ckpt = torch.load(CKPT_PATH, map_location=device)
decoder.load_state_dict(ckpt['decoder'])
if 'lora' in ckpt:
    extractor.load_state_dict(ckpt['lora'], strict=False)
    print("LoRA weights loaded.")
print("Checkpoint loaded successfully.")

# Final evaluation
print("--- Final Evaluation (2-tap LoRA) ---")
avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig = test_model_online_lora(
    extractor, decoder, val_loader, criterion, device
)

print(f"Final Model Benchmark (dinov3_lora_upgraded_2tap):")
print(f"Composite Loss: {avg_loss:.4f}")
print(f"KLD: {avg_kld:.4f}")
print(f"CC:  {avg_cc:.4f}")
print(f"SIM: {avg_sim:.4f}")
print(f"NSS: {avg_nss:.4f}")
print(f"AUC: {avg_auc:.4f}")
print(f"IG:  {avg_ig:.4f} bits (vs Gaussian center-bias baseline)")
