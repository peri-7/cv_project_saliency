# LoRA + Upgraded Decoder — Kaggle Smoke Test
# Runs 2 batches of 8 images with both USE_LORA=True and DECODER="upgraded"
# to verify the full ceiling configuration before committing to a 10-epoch run.
#
# What this checks:
#   1. LoraDinoV3ViT forward_intermediates API is compatible with this timm version
#   2. Gradient checkpointing + LoRA: the first-batch grad assertion in
#      train_one_epoch_lora passes (non-zero LoRA gradients)
#   3. ConvUpDecoder forward produces the expected output shape
#   4. train_one_epoch_lora + evaluate_model_online complete without error
#   5. test_model_online_lora (Gaussian IG baseline) completes without error
#   6. Checkpoint save/load round-trip works

!git clone https://github.com/peri-7/cv_project_saliency.git

import sys
import os
import torch

sys.path.append('/kaggle/working/cv_project_saliency/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cloud Hardware active: {device}")

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(token=UserSecretsClient().get_secret("HF_TOKEN"))

import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from src.dataset import LoraDataset
from src.decoder import ConvUpDecoder
from src.losses import Composite_Loss
from src.lora import LoraDinoV3ViT, train_one_epoch_lora, test_model_online_lora
from src.training_online import evaluate_model_online

print("--- SMOKE TEST: LoRA + ConvUpDecoder (ceiling config) ---")

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

full_train = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/train"),
    maps_dir=os.path.join(base_input_path, "maps/train"),
    fixations_dir=os.path.join(base_input_path, "fixations/train"),
    image_transform=image_transform,
    map_transform=map_transform
)
full_val = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/val"),
    maps_dir=os.path.join(base_input_path, "maps/val"),
    fixations_dir=os.path.join(base_input_path, "fixations/val"),
    image_transform=image_transform,
    map_transform=map_transform
)

# 16 images — two batches of 8 — enough to hit the first-batch grad check and
# exercise evaluate/test without waiting long.
train_loader = DataLoader(Subset(full_train, range(16)), batch_size=8, shuffle=False, num_workers=2)
val_loader   = DataLoader(Subset(full_val,  range(16)), batch_size=8, shuffle=False, num_workers=2)

# --- Build models ---
print("\n[1/6] Building LoraDinoV3ViT (grad_checkpointing=True)...")
extractor = LoraDinoV3ViT(grad_checkpointing=True).to(device)
lora_param_count = sum(p.numel() for p in extractor.trainable_parameters())
print(f"      LoRA trainable params: {lora_param_count:,}")

print("[2/6] Building ConvUpDecoder...")
decoder = ConvUpDecoder(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)
decoder_param_count = sum(p.numel() for p in decoder.parameters())
print(f"      Decoder params: {decoder_param_count:,}")

# Quick shape check: one forward pass before any training.
print("[3/6] Shape check (one forward pass, no grad)...")
with torch.no_grad():
    dummy = torch.randn(2, 3, 480, 640, device=device)
    feats = extractor(dummy)
    logits = decoder(list(feats.values()))
print(f"      extractor output shapes: {[tuple(v.shape) for v in feats.values()]}")
print(f"      decoder output shape:    {tuple(logits.shape)}  (expected [2, 1, 120, 160])")
assert logits.shape == (2, 1, 120, 160), f"Unexpected decoder shape: {logits.shape}"

criterion = Composite_Loss().to(device)
optimizer = optim.AdamW([
    {'params': decoder.parameters(), 'lr': 1e-4},
    {'params': extractor.trainable_parameters(), 'lr': 5e-5},
], weight_decay=1e-4)

# --- Train one epoch (triggers the first-batch grad assertion internally) ---
print("[4/6] train_one_epoch_lora (1 epoch, 16 images)...")
train_loss, kld, cc = train_one_epoch_lora(extractor, decoder, train_loader, optimizer, criterion, device)
print(f"      train loss={train_loss:.4f}  kld={kld:.4f}  cc={cc:.4f}")

# --- Validate ---
print("[5/6] evaluate_model_online...")
val_loss = evaluate_model_online(extractor, decoder, val_loader, criterion, device)
print(f"      val loss={val_loss:.4f}")

# --- Full 7-tuple test with Gaussian IG baseline ---
print("[6/6] test_model_online_lora (Gaussian center-bias IG)...")
loss, kld, cc, sim, nss, auc, ig = test_model_online_lora(extractor, decoder, val_loader, criterion, device)
print(f"      loss={loss:.4f}  kld={kld:.4f}  cc={cc:.4f}  sim={sim:.4f}")
print(f"      nss={nss:.4f}  auc={auc:.4f}  ig={ig:.4f} bits (Gaussian baseline)")

# --- Checkpoint round-trip ---
CKPT = "/kaggle/working/lora_smoke_ckpt.pth"
state = {
    'decoder': decoder.state_dict(),
    'lora': {k: v for k, v in extractor.state_dict().items() if 'lora_' in k},
}
torch.save(state, CKPT)
ckpt = torch.load(CKPT)
decoder.load_state_dict(ckpt['decoder'])
extractor.load_state_dict(ckpt['lora'], strict=False)
print("      Checkpoint save/load: OK")

print("\n--- SMOKE TEST PASSED — safe to run notebooks/lora.py ---")
