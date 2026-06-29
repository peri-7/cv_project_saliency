"""
Comparison of frozen backbones on saliency prediction metrics (SALICON).
Produces 4 different visualizations:
  1. Grouped bar chart - raw values, 2 subplots (loss/KLD vs the rest)
  2. Grouped bar chart - normalized [0,1], all metrics on one axis
  3. Radar / spider chart
  4. Heatmap

Requirements: pip install matplotlib pandas numpy
Current working directory: "cv_project_saliency/results/"
Visualizations are saved in the same directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# 1) DATA — edit here if needed
# ----------------------------------------------------------------------
data = {
    "Model":   ["ResNet", "ViT", "CLIP", "MAE", "DiNOv2", "DiNOv3", "SAM"],
    "Loss":    [1.5806, 1.0398, 0.6037, 0.8191, 0.5515, 0.5529, 0.7691],
    "KLD":     [0.2995, 0.2493, 0.2099, 0.2289, 0.2050, 0.2064, 0.2245],
    "CC":      [0.6824, 0.6951, 0.7156, 0.7025, 0.7121, 0.7257, 0.7051],
    "SIM":     [0.7321, 0.7579, 0.7794, 0.7676, 0.7862, 0.7853, 0.7707],
    "NSS":     [1.2975, 1.3117, 1.3519, 1.3200, 1.3447, 1.3712, 1.3255],
    "AUC":     [0.8555, 0.8638, 0.8686, 0.8668, 0.8694, 0.8697, 0.8677],
    "IG":      [0.7906, 0.8538, 0.9267, 0.8894, 0.9356, 0.9366, 0.8990],
}
df = pd.DataFrame(data).set_index("Model")

# Direction of each metric: True = "higher is better", False = "lower is better"
higher_is_better = {
    "Loss": False, "KLD": False, "CC": True, "SIM": True,
    "NSS": True, "AUC": True, "IG": True,
}
metrics = list(higher_is_better.keys())

# Fixed color per model (consistent across all charts)
palette = plt.cm.tab10(np.linspace(0, 1, len(df.index)))
colors = dict(zip(df.index, palette))


# ----------------------------------------------------------------------
# 2) GROUPED BAR CHART — raw values, in 2 subplots (different scales)
#    This  splits Loss/KLD (lower is better) from the rest (higher is better),
#    since otherwise the scales don't compare meaningfully on one axis.
# ----------------------------------------------------------------------
def plot_grouped_bars_raw():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 2.5]})

    n_models = len(df.index)
    width = 0.8 / n_models

    # -- Left subplot: Loss & KLD (lower is better) --
    ax = axes[0]
    left_metrics = ["Loss", "KLD"]
    x = np.arange(len(left_metrics))
    for i, model in enumerate(df.index):
        vals = df.loc[model, left_metrics].values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=model, color=colors[model])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\u2193" for m in left_metrics], fontsize=12)
    ax.set_title("Loss & KLD (lower is better)", fontsize=12)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)

    # -- Right subplot: CC, SIM, NSS, AUC, IG (higher is better) --
    ax = axes[1]
    right_metrics = ["CC", "SIM", "NSS", "AUC", "IG"]
    x = np.arange(len(right_metrics))
    for i, model in enumerate(df.index):
        vals = df.loc[model, right_metrics].values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=model, color=colors[model])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\u2191" for m in right_metrics], fontsize=12)
    ax.set_title("CC, SIM, NSS, AUC, IG (higher is better)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10,
              title="Backbone", frameon=False)

    fig.suptitle("Backbone comparison per metric (raw values)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("./01_grouped_bars_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 3) GROUPED BAR CHART — normalized [0,1], ALL metrics on one axis
#    This has the same vertical axis for all metrics, normalized so that
#    1.0 = best model on this metric, 0.0 = worst.
# ----------------------------------------------------------------------
def normalize_column(col, higher_better, visual_floor=0.06):
    """
    Min-max normalization to [0,1]: 1.0 = best model, 0.0 = worst.
    visual_floor gives the worst model a small height (e.g. 0.06)
    instead of exactly 0, ONLY so it stays visible in bar/radar charts -
    otherwise it visually disappears as if the bar were missing.
    Does not affect the heatmap (there the same function is used,
    but an exact value of 0 is fine/desired for the colormap).
    """
    lo, hi = col.min(), col.max()
    if hi == lo:
        return col * 0 + 1.0
    norm = (col - lo) / (hi - lo)
    if not higher_better:
        norm = 1 - norm
    return norm * (1 - visual_floor) + visual_floor


def plot_grouped_bars_normalized():
    norm_df = df.copy()
    for m in metrics:
        norm_df[m] = normalize_column(df[m], higher_is_better[m])

    fig, ax = plt.subplots(figsize=(13, 6.5))
    n_models = len(df.index)
    width = 0.8 / n_models
    x = np.arange(len(metrics))

    for i, model in enumerate(df.index):
        vals = norm_df.loc[model, metrics].values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=model, color=colors[model])

    arrows = {"Loss": "\u2193", "KLD": "\u2193", "CC": "\u2191", "SIM": "\u2191",
              "NSS": "\u2191", "AUC": "\u2191", "IG": "\u2191"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}{arrows[m]}" for m in metrics], fontsize=12)
    ax.set_ylabel("Normalized performance\n(1.0 = best model on this metric)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Backbone comparison — normalized performance per metric",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10,
              title="Backbone", frameon=False)
    fig.text(0.5, -0.02,
             "Note: the worst model on each metric gets a small visual height (not 0) so it stays visible.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig("./02_grouped_bars_normalized.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 4) RADAR / SPIDER CHART — very common in papers for exactly this use case
# ----------------------------------------------------------------------
def plot_radar():
    norm_df = df.copy()
    for m in metrics:
        norm_df[m] = normalize_column(df[m], higher_is_better[m])

    arrows = {"Loss": "\u2193", "KLD": "\u2193", "CC": "\u2191", "SIM": "\u2191",
              "NSS": "\u2191", "AUC": "\u2191", "IG": "\u2191"}
    labels = [f"{m}{arrows[m]}" for m in metrics]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

    for model in df.index:
        vals = norm_df.loc[model, metrics].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=model, color=colors[model])
        ax.fill(angles, vals, alpha=0.07, color=colors[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Radar chart — normalized backbone performance",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10,
              title="Backbone", frameon=False)
    fig.text(0.5, 0.0,
             "Note: the worst model on each metric gets a small visual radius (not 0) so it stays visible.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig("./03_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 5) HEATMAP — compact comparison, good for many models/metrics at once
# ----------------------------------------------------------------------
def plot_heatmap():
    norm_df = df.copy()
    for m in metrics:
        norm_df[m] = normalize_column(df[m], higher_is_better[m], visual_floor=0.0)

    arrows = {"Loss": "\u2193", "KLD": "\u2193", "CC": "\u2191", "SIM": "\u2191",
              "NSS": "\u2191", "AUC": "\u2191", "IG": "\u2191"}

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(norm_df[metrics].values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([f"{m}{arrows[m]}" for m in metrics], fontsize=12)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=12)

    # Annotate with the actual (raw) values
    for i, model in enumerate(df.index):
        for j, m in enumerate(metrics):
            raw_val = df.loc[model, m]
            text_color = "white" if norm_df.loc[model, m] < 0.3 or norm_df.loc[model, m] > 0.85 else "black"
            ax.text(j, i, f"{raw_val:.4f}", ha="center", va="center",
                     fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Normalized performance (green = better)", fontsize=10)
    ax.set_title("Backbone comparison heatmap (SALICON, frozen backbone + shared decoder)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig("./04_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_grouped_bars_raw()
    plot_grouped_bars_normalized()
    plot_radar()
    plot_heatmap()
    print("All charts saved to cv_project_saliency/results/")
