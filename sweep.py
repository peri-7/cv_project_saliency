# Decoder-width sweep: find the hidden_dim where the backbone comparison is most
# legible (metrics plateau / backbone separation peaks), BEFORE locking the decoder
# config for the whole roster.
#
# CHEAP BY DESIGN:
#   * You ALREADY have the 128 result -- your existing benchmark run IS hidden_dim=128,
#     so it is NOT retrained here.
#   * Test the extremes {256, 512} first and bisect: if 512 ~= 256 the plateau is
#     at/below 256 -> lock 256; only if 512 >> 256 do you probe 384/768.
#   * EPOCHS is cut to 5 -- this only needs the *relative* ordering of widths, not
#     converged numbers. Retrain just the winning width at 10 epochs for the real
#     benchmark.
#
# Run for ResNet first (fast). Then flip BACKBONE = "sam" and re-run at the winning
# width to confirm the SAM-vs-ResNet gap actually widens there. Everything else is held
# identical to the real notebooks (Adam 1e-4, LinearLR 1.0->0.01, batch 16, val proxy,
# uniform IG baseline) so numbers stay comparable to your existing runs.

# 1. Clone your repository directly into the Kaggle working directory
get_ipython().system('git clone https://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@github.com/peri-7/cv_project_saliency.git')

import sys
import os
import torch

sys.path.append('/kaggle/working/cv_project_saliency/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cloud Hardware active: {device}")

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from src.dataset import LoraDataset
from src.models import ResNet, SamViT
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.training_online import train_one_epoch_online, evaluate_model_online, test_model_online

# ---------------------------------------------------------------------------
# Knobs for the sweep (see cheap-by-design note at top)
# ---------------------------------------------------------------------------
BACKBONE = "resnet"            # "resnet" (fast, first pass) or "sam"
HIDDEN_DIMS = [256, 512]       # extremes first; add 384 only if 512 clearly beats 256
EPOCHS = 5                     # probe depth; bump to 10 for the final locked-width run
SEED = 0                       # re-seeded before each decoder init for fair comparison

# ---------------------------------------------------------------------------
# Data + frozen backbone are built ONCE and reused across every hidden_dim.
# The backbone is frozen and identical, so re-instantiating it per width would just
# waste compute.
# ---------------------------------------------------------------------------
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

train_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/train"),
    maps_dir=os.path.join(base_input_path, "maps/train"),
    fixations_dir=os.path.join(base_input_path, "fixations/train"),
    image_transform=image_transform,
    map_transform=map_transform
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

val_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, "images/images/val"),
    maps_dir=os.path.join(base_input_path, "maps/val"),
    fixations_dir=os.path.join(base_input_path, "fixations/val"),
    image_transform=image_transform,
    map_transform=map_transform
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

if BACKBONE == "resnet":
    extractor = ResNet().to(device)
elif BACKBONE == "sam":
    extractor = SamViT().to(device)
else:
    raise ValueError(f"Unknown BACKBONE={BACKBONE!r} (expected 'resnet' or 'sam')")

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
results = {}   # hidden_dim -> 7-tuple (loss, kld, cc, sim, nss, auc, ig)

for hidden_dim in HIDDEN_DIMS:
    print("=" * 60)
    print(f"--- {BACKBONE.upper()} | hidden_dim = {hidden_dim} ---")

    # Re-seed so the only difference across widths is the width itself (not RNG).
    torch.manual_seed(SEED)

    decoder = Decoder(in_channels_list=extractor.out_channels, hidden_dim=hidden_dim).to(device)
    criterion = Composite_Loss().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=EPOCHS)

    ckpt_path = f"/kaggle/working/sweep_{BACKBONE}_h{hidden_dim}.pth"
    patience = 0
    val_min = float('inf')

    for epoch in range(EPOCHS):
        current_lr = scheduler.get_last_lr()[0]

        train_loss, kld, cc = train_one_epoch_online(extractor, decoder, train_loader, optimizer, criterion, device)
        val_loss = evaluate_model_online(extractor, decoder, val_loader, criterion, device)

        print(f"  Epoch {epoch+1:2d} | LR: {current_lr:.3e} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        scheduler.step()

        if val_loss < val_min:
            patience = 0
            val_min = val_loss
            torch.save(decoder.state_dict(), ckpt_path)
        else:
            patience += 1

        if patience > 3:
            print(f"  Early stop on epoch {epoch+1}.")
            break

    # Benchmark the BEST checkpoint for this width.
    decoder.load_state_dict(torch.load(ckpt_path))
    metrics = test_model_online(extractor, decoder, val_loader, criterion, device)
    results[hidden_dim] = metrics

    avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig = metrics
    print(f"  -> Loss {avg_loss:.4f} | KLD {avg_kld:.4f} | CC {avg_cc:.4f} | "
          f"SIM {avg_sim:.4f} | NSS {avg_nss:.4f} | AUC {avg_auc:.4f} | IG {avg_ig:.4f}")

# ---------------------------------------------------------------------------
# Summary table -- eyeball where metrics plateau. Compare against your existing
# 128 benchmark row by hand: if 256 barely beats 128, the decoder was never the
# ceiling; if 256 >> 128 but 512 ~= 256, lock 256.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"SWEEP SUMMARY ({BACKBONE.upper()})  --  compare vs your existing hidden_dim=128 run")
print("=" * 78)
header = f"{'hidden_dim':>10} | {'Loss':>7} | {'KLD':>6} | {'CC':>6} | {'SIM':>6} | {'NSS':>6} | {'AUC':>6} | {'IG':>6}"
print(header)
print("-" * len(header))
for hidden_dim in HIDDEN_DIMS:
    if hidden_dim not in results:
        continue
    loss, kld, cc, sim, nss, auc, ig = results[hidden_dim]
    print(f"{hidden_dim:>10} | {loss:>7.4f} | {kld:>6.4f} | {cc:>6.4f} | "
          f"{sim:>6.4f} | {nss:>6.4f} | {auc:>6.4f} | {ig:>6.4f}")
