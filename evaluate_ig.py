"""
evaluate_ig.py
--------------
Re-evaluates ONLY the Information Gain (IG) metric for all trained backbones,
using the empirical center-bias baseline computed from the training fixations.

Prerequisites:
    - ./saved_models/center_bias_baseline.pt  (from compute_baseline.py)
    - ./saved_models/best_*_decoder.pth       (all trained checkpoints)

Usage:
    python evaluate_ig.py
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from src.dataset import LoraDataset
from src.decoder import Decoder
from src.metrics import information_gain
from src.models import ResNet, SamViT, ViT, MaeViT, DinoV2ViT, DinoV3ViT, ClipViT


# ── Configuration ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BASE_INPUT_PATH  = '/home/stamata/Downloads/salicon'
BASELINE_PATH    = './saved_models/center_bias_baseline.pt'
MODELS_DIR       = './saved_models'

# Each entry: (display_name, extractor_class, hidden_dim, checkpoint_filename)
BACKBONES = [
    ("ResNet-50",   ResNet,     256, "best_resnet_decoder.pth"),
    ("SAM ViT-B",   SamViT,     256, "best_sam_decoder.pth"),
    ("ViT-B",       ViT,        256, "best_vit_decoder.pth"),
    ("MAE ViT-B",   MaeViT,     256, "best_mae_decoder.pth"),
    ("DINOv2 ViT-B",DinoV2ViT,  256, "best_dinov2_decoder.pth"),
    ("DINOv3 ViT-B",DinoV3ViT,  256, "best_dinov3_decoder.pth"),
    ("CLIP ViT-B",  ClipViT,    256, "best_clip_decoder.pth"),
]


# ── Transforms (identical to training) ────────────────────────────────────────

image_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

map_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])


# ── Dataset ───────────────────────────────────────────────────────────────────

val_dataset = LoraDataset(
    image_dir=os.path.join(BASE_INPUT_PATH, "images/images/val"),
    maps_dir=os.path.join(BASE_INPUT_PATH, "maps/val"),
    fixations_dir=os.path.join(BASE_INPUT_PATH, "fixations/val"),
    image_transform=image_transform,
    map_transform=map_transform
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)


# ── Load Empirical Baseline ───────────────────────────────────────────────────

print(f"Loading empirical baseline from: {BASELINE_PATH}")
baseline = torch.load(BASELINE_PATH, map_location=device)  # [1, 1, 480, 640]
print(f"Baseline loaded. Shape: {baseline.shape}, Sum: {baseline.sum().item():.6f}\n")


# ── IG Evaluation Function ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ig(extractor, decoder, dataloader, baseline_prob, device):
    extractor.eval()
    decoder.eval()

    total_ig = 0.0

    for images, target_map, fixation_map in tqdm(dataloader, leave=False):
        images      = images.to(device)
        target_map  = target_map.to(device)
        fixation_map = fixation_map.to(device)

        b, c, target_height, target_width = target_map.shape

        # Forward pass
        features_dict = extractor(images)
        features = list(features_dict.values())
        raw_logits = decoder(features)

        # Upsample to ground truth size
        matched_logits = F.interpolate(
            raw_logits,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        # Convert to probability distribution (required by IG)
        pred_prob = F.softmax(
            matched_logits.view(b, 1, -1), dim=2
        ).view(b, 1, target_height, target_width)

        # Expand baseline to match batch size
        batch_baseline = baseline_prob.expand(b, -1, -1, -1).to(device)

        total_ig += information_gain(pred_prob, fixation_map, batch_baseline)

    return total_ig / len(dataloader)


# ── Main Loop ─────────────────────────────────────────────────────────────────

print("=" * 55)
print(f"{'Backbone':<20} {'IG (empirical baseline)':>25}")
print("=" * 55)

results = []

for name, ExtractorClass, hidden_dim, ckpt_file in BACKBONES:

    ckpt_path = os.path.join(MODELS_DIR, ckpt_file)

    if not os.path.exists(ckpt_path):
        print(f"{name:<20} {'[checkpoint not found]':>25}")
        continue

    # Build model
    extractor = ExtractorClass().to(device)
    decoder   = Decoder(in_channels_list=extractor.out_channels, hidden_dim=hidden_dim).to(device)
    decoder.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Evaluate
    print(f"Evaluating {name}...")
    avg_ig = evaluate_ig(extractor, decoder, val_loader, baseline, device)
    results.append((name, avg_ig))

    print(f"{name:<20} {avg_ig:>25.4f} bits")

    # Free GPU memory before loading next backbone
    del extractor, decoder
    torch.cuda.empty_cache()

print("=" * 55)
print("\nDone.")