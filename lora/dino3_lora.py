# LoRA + Upgraded Decoder experiment (see lora/LORA.md).
# NOT part of the frozen-backbone fairness study — this run chases the best
# possible SALICON score with DINOv3 ViT-B. The two upgrades are independently
# switchable for the ablation grid:
#
#   | Run      | USE_LORA | DECODER  | Isolates                            |
#   |----------|----------|----------|-------------------------------------|
#   | Ceiling  | True     | upgraded | both upgrades together              |
#   | -LoRA    | False    | upgraded | decoder-only gain vs frozen dinov3  |
#   | -decoder | True     | baseline | LoRA-only gain vs frozen dinov3     |
#   | baseline | False    | baseline | == existing notebooks/dinov3.py     |

USE_LORA = False         # True -> LoraDinoV3ViT + train_one_epoch_lora; False -> frozen DinoV3ViT + train_one_epoch_online
DECODER  = "upgraded"    # "upgraded" -> ConvUpDecoder; "baseline" -> Decoder

# Resuming an interrupted run (Kaggle time limit):
#   1. After the session dies, grab /kaggle/working/resume_<RUN_NAME>.pth from
#      the session output and publish it as a Kaggle dataset.
#   2. Mount that dataset as an input in the next session and point RESUME_FROM
#      at the file. Leave None for a fresh run.
# The resume file carries FULL state — current + best weights, optimizer
# moments, scheduler, epoch, patience, val_min — so the resumed run continues
# exactly where it left off (no AdamW momentum reset).
RESUME_FROM = None  # e.g. "/kaggle/input/lora-resume/resume_dinov3_lora_upgraded.pth"

# 1. Clone your repository into the Kaggle working directory.
#    Script-safe: works whether this file is executed as a plain `python` script
#    or pasted into a notebook cell (the old `!git clone` magic only worked in a
#    notebook and raised SyntaxError under `%run`/`python`). Idempotent — skips
#    the clone if the repo is already present.
import sys
import os
import subprocess

_REPO_DIR = '/kaggle/working/cv_project_saliency'
if not os.path.isdir(_REPO_DIR):
    subprocess.run(
        ['git', 'clone', 'https://github.com/peri-7/cv_project_saliency.git', _REPO_DIR],
        check=True,
    )

import torch

# 2. Append the cloned repository folder to Python's path
sys.path.append('/kaggle/working/cv_project_saliency/')

# 3. Verify hardware acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cloud Hardware active: {device}")

# 4. DINOv3 weights are license-gated on Hugging Face — authenticate BEFORE
#    building the extractor. Token resolution order:
#      1. HF_TOKEN environment variable — works when this file is launched as a
#         script, e.g.  HF_TOKEN="hf_..." python lora/dino3_lora.py
#      2. Kaggle secret named HF_TOKEN — for running inside a notebook cell.
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        raise RuntimeError(
            "No Hugging Face token found. Either launch with "
            "`HF_TOKEN=... python lora/dino3_lora.py`, or add a Kaggle secret "
            "named HF_TOKEN."
        ) from e
login(token=hf_token)

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from tqdm import tqdm

from src.dataset import LoraDataset
from src.models import DinoV3ViT
from src.decoder import Decoder, ConvUpDecoder
from src.losses import Composite_Loss
from src.lora import LoraDinoV3ViT, train_one_epoch_lora
from src.training_online import train_one_epoch_online, evaluate_model_online, test_model_online

RUN_NAME = f"dinov3_{'lora' if USE_LORA else 'frozen'}_{DECODER}"
CKPT_PATH = f"/kaggle/working/best_{RUN_NAME}.pth"
RESUME_PATH = f"/kaggle/working/resume_{RUN_NAME}.pth"
print(f"--- LoRA experiment run: {RUN_NAME} ---")

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

base_input_path = '/kaggle/input/datasets/roshan401/salicon'

# Smaller batch when LoRA is on: backprop through the trunk needs activation
# memory even with gradient checkpointing (8 fits a 16 GB T4/P100).
batch_size = 8 if USE_LORA else 16

train_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/train"),
    maps_dir=os.path.join(base_input_path, "maps/train"),
    fixations_dir=os.path.join(base_input_path, "fixations/train"),
    image_transform=image_transform,
    map_transform=map_transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

val_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/val"),
    maps_dir=os.path.join(base_input_path, "maps/val"),
    fixations_dir=os.path.join(base_input_path, "fixations/val"),
    image_transform=image_transform,
    map_transform=map_transform
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- Extractor: LoRA-adapted or frozen DINOv3 ---
if USE_LORA:
    extractor = LoraDinoV3ViT(tap_blocks=[2, 5, 8, 11], grad_checkpointing=True).to(device)
    train_fn = train_one_epoch_lora
else:
    extractor = DinoV3ViT(tap_blocks=[2, 5, 8, 11]).to(device)
    train_fn = train_one_epoch_online

# --- Decoder: upgraded progressive-upsampling or baseline ---
# hidden_dim locked at 256 per project convention.
decoder_cls = ConvUpDecoder if DECODER == "upgraded" else Decoder
decoder = decoder_cls(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)

criterion = Composite_Loss().to(device)

# AdamW for both runs (decision #6 in lora/LORA.md). With LoRA, two param
# groups: the decoder at the usual lr, the adapters slightly lower.
if USE_LORA:
    lora_params = extractor.trainable_parameters()
    print(f"Trainable params — decoder: {sum(p.numel() for p in decoder.parameters()):,} | "
          f"LoRA: {sum(p.numel() for p in lora_params):,}")
    optimizer = optim.AdamW([
        {'params': decoder.parameters(), 'lr': 1e-4},
        {'params': lora_params, 'lr': 5e-5},
    ], weight_decay=1e-4)
else:
    optimizer = optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 10
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)


def save_checkpoint(path):
    state = {'decoder': decoder.state_dict()}
    if USE_LORA:
        state['lora'] = {k: v for k, v in extractor.state_dict().items() if 'lora_' in k}
    torch.save(state, path)


def load_checkpoint(path):
    ckpt = torch.load(path)
    decoder.load_state_dict(ckpt['decoder'])
    if USE_LORA:
        extractor.load_state_dict(ckpt['lora'], strict=False)


def save_resume_checkpoint(path, next_epoch):
    """Full training state, written EVERY epoch so a killed session loses at
    most the epoch in flight. Embeds the best-so-far weights too, so resuming
    needs only this one file."""
    state = {
        'run_name': RUN_NAME,
        'epoch': next_epoch,
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_min': val_min,
        'patience': patience,
        'best': torch.load(CKPT_PATH),
    }
    if USE_LORA:
        state['lora'] = {k: v for k, v in extractor.state_dict().items() if 'lora_' in k}
    torch.save(state, path)


# Training Loop with Early Stopping
start_epoch = 0
patience = 0
val_min = float('inf')

if RESUME_FROM is not None:
    ckpt = torch.load(RESUME_FROM, map_location=device)
    assert ckpt['run_name'] == RUN_NAME, (
        f"Resume checkpoint is for run {ckpt['run_name']!r} but the flags "
        f"select {RUN_NAME!r} — refusing to mix configurations."
    )
    decoder.load_state_dict(ckpt['decoder'])
    if USE_LORA:
        extractor.load_state_dict(ckpt['lora'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch']
    val_min = ckpt['val_min']
    patience = ckpt['patience']
    # Restore the best-so-far weights to the working dir so best-checkpoint
    # tracking and the final evaluation work exactly as in a single session.
    torch.save(ckpt['best'], CKPT_PATH)
    print(f"Resumed from {RESUME_FROM}: continuing at epoch {start_epoch+1}/{epochs} "
          f"(val_min={val_min:.4f}, patience={patience})")

for epoch in range(start_epoch, epochs):

    current_lr = scheduler.get_last_lr()[0]

    train_loss, kld, cc = train_fn(extractor, decoder, train_loader, optimizer, criterion, device)
    val_loss = evaluate_model_online(extractor, decoder, val_loader, criterion, device)

    print(f"Epoch {epoch+1} | LR: {current_lr:.3e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    scheduler.step()

    # Save the absolute best weights to the hard drive
    if val_loss < val_min:
        patience = 0
        val_min = val_loss
        save_checkpoint(CKPT_PATH)
        print("  -> New best model saved!")
    else:
        patience += 1

    # Full-state resume checkpoint, overwritten every epoch (see RESUME_FROM).
    save_resume_checkpoint(RESUME_PATH, next_epoch=epoch + 1)

    if patience > 3:
        print(f"Early Stopping triggered on Epoch {epoch}. Restoring best weights.")
        load_checkpoint(CKPT_PATH)
        break

print("-" * 50)
print("Training complete.")

print("--- Final Evaluation ---")

# Always benchmark the BEST checkpoint, not whatever weights happen to be in
# memory after the last epoch.
load_checkpoint(CKPT_PATH)

# --- Empirical center-bias baseline for Information Gain ---
# IG here is measured against the EMPIRICAL center-bias baseline (the actual
# fixation density accumulated over the SALICON training set), NOT the analytic
# Gaussian. This is the same reference the frozen-roster empirical-IG passes use
# (see notebooks/evaluate_dinov3_781011.py), so this IG IS comparable to those.
import numpy as np
import scipy.io as sio
from compute_baseline import parse_fixations  # reused fixation .mat parser

BASELINE_PATH = "/kaggle/working/center_bias_baseline.pt"
HEIGHT, WIDTH = 480, 640


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

# Run the strict metric calculation on the validation proxy set. The six
# baseline-independent metrics are identical to the roster; IG uses the
# empirical center-bias baseline above.
avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig = test_model_online(
    extractor, decoder, val_loader, criterion, device, baseline_prob=baseline)

print(f"Final Model Benchmark ({RUN_NAME}):")
print(f"Composite Loss: {avg_loss:.4f}")
print(f"KLD: {avg_kld:.4f}")
print(f"CC:  {avg_cc:.4f}")
print(f"SIM: {avg_sim:.4f}")
print(f"NSS: {avg_nss:.4f}")
print(f"AUC: {avg_auc:.4f}")
print(f"IG:  {avg_ig:.4f} bits (vs empirical center-bias baseline)")
