# Re-evaluates the already-trained DINOv3 decoder with tap_blocks [7, 8, 10, 11]
# and reports the full metric row, including the IG that the original run missed
# (empirical baseline).
#
# It doesn't retrain: it loads best_dinov3_decoder_781011.pth and runs only the
# final evaluation pass, matching how notebooks/dinov3.py produced the original
# numbers (test_model_online, uniform IG), plus one extra pass against the
# empirical center-bias baseline for the last column.
#
# Prerequisites on Kaggle:
#   1. Repo cloned to /kaggle/working/cv_project_saliency/
#   2. SALICON dataset at /kaggle/input/datasets/roshan401/salicon
#   3. best_dinov3_decoder_781011.pth reachable at one of the CKPT_CANDIDATES
#      paths below (upload it as a Kaggle dataset or drop it in /kaggle/working).
#
# The empirical baseline (center_bias_baseline.pt) is loaded if present, else
# recomputed from the training fixations and cached.

import sys
import os
import torch
import numpy as np
import scipy.io as sio
from tqdm import tqdm

sys.path.append('/kaggle/working/cv_project_saliency/')

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.dataset import LoraDataset
from src.models import DinoV3ViT
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.training_online import test_model_online
from scripts.compute_baseline import parse_fixations  # reused fixation .mat parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Configuration ─────────────────────────────────────────────────────────────

TAP_BLOCKS   = [7, 8, 10, 11]
HEIGHT, WIDTH = 480, 640

base_input_path = '/kaggle/input/datasets/roshan401/salicon'

# First existing path wins. Add/adjust as needed for your Kaggle layout.
CKPT_CANDIDATES = [
    '/kaggle/working/best_dinov3_decoder_781011.pth',
    '/kaggle/working/cv_project_saliency/saved_models/best_dinov3_decoder_781011.pth',
    '/kaggle/input/dinov3-781011/best_dinov3_decoder_781011.pth',
]

BASELINE_PATH = '/kaggle/working/center_bias_baseline.pt'


# ── Transforms (identical to training) ────────────────────────────────────────

image_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

map_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor()
])


# ── Datasets ──────────────────────────────────────────────────────────────────

val_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/val"),
    maps_dir=os.path.join(base_input_path, "maps/val"),
    fixations_dir=os.path.join(base_input_path, "fixations/val"),
    image_transform=image_transform,
    map_transform=map_transform
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)


# ── Empirical center-bias baseline (load or recompute) ────────────────────────

def build_empirical_baseline(fixations_dir, height, width):
    """Accumulate all training fixations into a normalized [1,1,H,W] prior."""
    mat_files = sorted(f for f in os.listdir(fixations_dir) if f.endswith('.mat'))
    if not mat_files:
        raise FileNotFoundError(f"No .mat fixation files in {fixations_dir}")
    print(f"Recomputing empirical baseline from {len(mat_files)} training fixations...")
    acc = np.zeros((height, width), dtype=np.float64)
    for fname in tqdm(mat_files):
        mat_data = sio.loadmat(os.path.join(fixations_dir, fname))
        acc += parse_fixations(mat_data, height, width)
    acc /= acc.sum()
    return torch.tensor(acc, dtype=torch.float32).view(1, 1, height, width)


if os.path.exists(BASELINE_PATH):
    print(f"Loading empirical baseline from: {BASELINE_PATH}")
    baseline = torch.load(BASELINE_PATH, map_location=device)
else:
    baseline = build_empirical_baseline(
        os.path.join(base_input_path, "fixations/train"), HEIGHT, WIDTH
    )
    torch.save(baseline, BASELINE_PATH)
    print(f"Empirical baseline saved to: {BASELINE_PATH}")

baseline = baseline.to(device)
print(f"Baseline shape: {tuple(baseline.shape)}, sum: {baseline.sum().item():.6f}\n")


# ── Model ─────────────────────────────────────────────────────────────────────

ckpt_path = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
if ckpt_path is None:
    raise FileNotFoundError(
        "best_dinov3_decoder_781011.pth not found. Checked:\n  "
        + "\n  ".join(CKPT_CANDIDATES)
    )
print(f"Loading checkpoint: {ckpt_path}")

extractor = DinoV3ViT(tap_blocks=TAP_BLOCKS).to(device)
decoder = Decoder(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)
decoder.load_state_dict(torch.load(ckpt_path, map_location=device))

criterion = Composite_Loss().to(device)


# ── Evaluation ────────────────────────────────────────────────────────────────
# Pass 1 (uniform IG): reproduces cols 1-7 exactly as the original table row.
# Pass 2 (empirical baseline): supplies the missing IG (empirical) column.

print("--- Phase 3: Final Evaluation (DINOv3 ViT-B, taps [7, 8, 10, 11]) ---")

avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig_uniform = \
    test_model_online(extractor, decoder, val_loader, criterion, device)

# Only the IG value differs in this pass; the other six are baseline-independent.
_, _, _, _, _, _, avg_ig_empirical = \
    test_model_online(extractor, decoder, val_loader, criterion, device,
                      baseline_prob=baseline)

print(f"\nFinal Model Benchmark (DINOv3 ViT-B, taps [7, 8, 10, 11]):")
print(f"Composite Loss:        {avg_loss:.4f}")
print(f"KLD:                   {avg_kld:.4f}")
print(f"CC:                    {avg_cc:.4f}")
print(f"SIM:                   {avg_sim:.4f}")
print(f"NSS:                   {avg_nss:.4f}")
print(f"AUC:                   {avg_auc:.4f}")
print(f"IG (uniform):          {avg_ig_uniform:.4f} bits")
print(f"IG (empirical):        {avg_ig_empirical:.4f} bits   <-- the missing value")
