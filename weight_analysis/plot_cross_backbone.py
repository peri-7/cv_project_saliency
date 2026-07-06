"""Cross-backbone feature-depth figures for the Weight Analysis section.

Reads the committed CSVs (no torch/data needed) and writes:
  depth_importance_A.png - Analysis A: per-tap importance share (%) per backbone.
  depth_ablation_B.png   - Analysis B: delta-loss when each tap is zeroed.

Backbones share one axis so the depth gradient is directly comparable. ResNet is
the confounded baseline (its taps differ in channels and resolution), so compare
the /16 ViTs to each other.

Run: python weight_analysis/plot_cross_backbone.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

# display name -> csv stem. ViT family shares the 4-block layout; ResNet is stages.
BACKBONES = ["dinov3", "vit", "clip", "sam", "mae", "resnet"]
NICE = {"dinov3": "DINOv3", "vit": "ViT (sup.)", "clip": "CLIP",
        "sam": "SAM", "mae": "MAE", "resnet": "ResNet*"}
DEPTH_LABELS = ["shallow", "early-mid", "late-mid", "deep"]   # tap 1..4 generic


def read_column_sums(stem):
    path = os.path.join(HERE, f"{stem}_tap_importance.csv")
    with open(path) as f:
        rows = list(csv.reader(f))
    for r in rows:
        if r and r[0] == "column_sum":
            return np.array([float(x) for x in r[1:]])
    raise ValueError(f"no column_sum in {path}")


def read_ablation_deltaloss(stem):
    path = os.path.join(HERE, f"{stem}_ablation.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # order: the 4 "zero ..." rows, in file order (shallow->deep)
    return np.array([float(r["delta_loss"]) for r in rows if r["row"].startswith("zero")])


# Figure 1: Analysis A, per-tap importance share.
def plot_A():
    shares = {}
    for b in BACKBONES:
        cs = read_column_sums(b)
        shares[b] = 100.0 * cs / cs.sum()
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.13
    for i, b in enumerate(BACKBONES):
        ax.bar(x + (i - 2.5) * width, shares[b], width, label=NICE[b])
    ax.set_xticks(x)
    ax.set_xticklabels([f"tap {k+1}\n({DEPTH_LABELS[k]})" for k in range(4)])
    ax.set_ylabel("Analysis A importance share (%)")
    ax.set_title("Where the decoder reads from (Analysis A: |W|·std, per-tap share)")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "depth_importance_A.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)
    for b in BACKBONES:
        print(f"  {NICE[b]:12s} shares %: " + "  ".join(f"{v:5.1f}" for v in shares[b]))


# Figure 2: Analysis B, causal delta-loss.
def plot_B():
    dl = {b: read_ablation_deltaloss(b) for b in BACKBONES}
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in BACKBONES:
        ax.plot(x, dl[b], marker="o", label=NICE[b])
    ax.set_xticks(x)
    ax.set_xticklabels([f"tap {k+1}\n({DEPTH_LABELS[k]})" for k in range(4)])
    ax.set_ylabel(r"$\Delta$loss when tap is zeroed (higher = more load-bearing)")
    ax.set_title("Causal tap importance (Analysis B: leave-one-tap-out)")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "depth_ablation_B.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)
    for b in BACKBONES:
        print(f"  {NICE[b]:12s} dloss:  " + "  ".join(f"{v:5.2f}" for v in dl[b]))


if __name__ == "__main__":
    print("=== Analysis A (importance share %) ===")
    plot_A()
    print("\n=== Analysis B (delta-loss) ===")
    plot_B()
