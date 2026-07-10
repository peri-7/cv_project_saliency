# Evaluation-only pass for the DINOv3 + LoRA / baseline-decoder run
# (dino3_lora.py with USE_LORA=True, DECODER="baseline", taps [2,5,8,11]).
#
# Handy when the training session saved a checkpoint but Kaggle's 12h limit killed
# the process before the final evaluation block ran. It doesn't retrain: it loads
# the checkpoint and runs the same final evaluation dino3_lora.py would have -- the
# six baseline-independent metrics plus Information Gain against the empirical
# center-bias baseline (from the SALICON training fixations), not the Gaussian one.
#
# Prerequisites on Kaggle (GPU + Internet on):
#   1. Repo at /kaggle/working/cv_project_saliency (cloned below if missing).
#   2. SALICON dataset at /kaggle/input/datasets/roshan401/salicon.
#   3. HF token: run as `HF_TOKEN=... python .../dino3_lora_eval.py` or add a
#      Kaggle secret named HF_TOKEN (DINOv3 weights are license-gated).
#   4. The trained checkpoints ship inside the repo under lora/, so they arrive
#      with the clone -- no separate Kaggle dataset needed for the weights.

# Config: must match the training run being evaluated.
TAP_BLOCKS = [2, 5, 8, 11]   # LoRA + baseline decoder run tapped these blocks
HIDDEN_DIM = 256             # baseline Decoder width (locked project convention)
HEIGHT, WIDTH = 480, 640

# 1. Clone the repo into the Kaggle working dir (script-safe + idempotent).
import sys
import os
import subprocess

_REPO_DIR = '/kaggle/working/cv_project_saliency'
if not os.path.isdir(_REPO_DIR):
    subprocess.run(
        ['git', 'clone', 'https://github.com/peri-7/cv_project_saliency.git', _REPO_DIR],
        check=True,
    )
sys.path.append(_REPO_DIR + '/')

import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cloud Hardware active: {device}")

# 2. DINOv3 weights are license-gated on Hugging Face — authenticate before
#    building the extractor. HF_TOKEN env var first (script launch), then the
#    Kaggle secret named HF_TOKEN (notebook cell).
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        raise RuntimeError(
            "No Hugging Face token found. Either launch with "
            "`HF_TOKEN=... python lora/dino3_lora_eval.py`, or add a Kaggle "
            "secret named HF_TOKEN."
        ) from e
login(token=hf_token)

from src.dataset import LoraDataset
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.lora import LoraDinoV3ViT
from src.training_online import test_model_online
from scripts.compute_baseline import parse_fixations  # reused fixation .mat parser

base_input_path = '/kaggle/input/datasets/roshan401/salicon'

# Checkpoints are shipped INSIDE the git repo (committed under lora/), so they
# arrive with the clone above — no Kaggle dataset needed for the weights. First
# existing path wins: prefer the standalone best checkpoint; fall back to the
# resume checkpoint, from which we pull the embedded best weights.
_REPO_LORA = os.path.join(_REPO_DIR, 'lora')
CKPT_CANDIDATES = [
    os.path.join(_REPO_LORA, 'best_dinov3_lora_simpledecoder.pth'),
    os.path.join(_REPO_LORA, 'resume_dinov3_lora_baseline.pth'),
]

BASELINE_PATH = '/kaggle/working/center_bias_baseline.pt'


# --- Transforms (identical to training) ---
image_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
map_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor()
])

val_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/val"),
    maps_dir=os.path.join(base_input_path, "maps/val"),
    fixations_dir=os.path.join(base_input_path, "fixations/val"),
    image_transform=image_transform,
    map_transform=map_transform
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)


# --- Empirical center-bias baseline (load cached, else recompute + cache) ---
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


# --- Rebuild the model and load the trained weights ---
ckpt_path = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
if ckpt_path is None:
    raise FileNotFoundError(
        "Checkpoint not found. Checked:\n  " + "\n  ".join(CKPT_CANDIDATES)
    )
print(f"Loading checkpoint: {ckpt_path}")

# Same LoRA architecture as training (all 12 blocks, r=8, qkv+proj by default);
# only tap_blocks was overridden. grad_checkpointing not needed for inference.
extractor = LoraDinoV3ViT(tap_blocks=TAP_BLOCKS).to(device)
decoder = Decoder(in_channels_list=extractor.out_channels, hidden_dim=HIDDEN_DIM).to(device)

ckpt = torch.load(ckpt_path, map_location=device)


def extract_best_weights(ckpt):
    """Pull (decoder_state, lora_state) out of either checkpoint format:
      - best checkpoint (save_checkpoint):        {'decoder', 'lora'}
      - resume checkpoint (save_resume_checkpoint): full state that ALSO embeds
        the best-so-far weights under 'best' — we prefer those over the
        last-epoch 'decoder'/'lora' also present in the file.
    """
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint is not a dict (got {type(ckpt)}).")
    if 'best' in ckpt and isinstance(ckpt['best'], dict):
        best = ckpt['best']
        if 'decoder' in best and 'lora' in best:
            print("  Using embedded best weights from resume checkpoint.")
            return best['decoder'], best['lora']
    if 'decoder' in ckpt and 'lora' in ckpt:
        return ckpt['decoder'], ckpt['lora']
    raise ValueError(
        "Checkpoint has neither a best {'decoder','lora'} pair nor top-level "
        f"'decoder'/'lora'. Keys: {list(ckpt.keys())}"
    )


decoder_state, lora_state = extract_best_weights(ckpt)
decoder.load_state_dict(decoder_state)
# strict=False: lora_state holds only the LoRA adapter tensors; the frozen
# DINOv3 base weights stay as loaded from the pretrained checkpoint.
extractor.load_state_dict(lora_state, strict=False)

criterion = Composite_Loss().to(device)


# --- Final evaluation (IG vs empirical center-bias baseline) ---
print("--- Final Evaluation (DINOv3 + LoRA, taps [2, 5, 8, 11], baseline decoder) ---")

avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig = test_model_online(
    extractor, decoder, val_loader, criterion, device, baseline_prob=baseline)

print(f"\nFinal Model Benchmark (dinov3_lora_baseline):")
print(f"Composite Loss: {avg_loss:.4f}")
print(f"KLD: {avg_kld:.4f}")
print(f"CC:  {avg_cc:.4f}")
print(f"SIM: {avg_sim:.4f}")
print(f"NSS: {avg_nss:.4f}")
print(f"AUC: {avg_auc:.4f}")
print(f"IG:  {avg_ig:.4f} bits (vs empirical center-bias baseline)")
