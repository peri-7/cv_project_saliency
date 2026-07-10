"""
Backbone comparison figures on the SALICON saliency metrics.

Produces two figures, saved in the current working directory:
  02_grouped_bars_normalized.png - all metrics on one [0,1] axis
  04_heatmap.png                 - compact model x metric grid

Run from cv_project_saliency/results/.  Needs matplotlib, pandas, numpy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Final Phase I scores. Edit here to regenerate the figures.
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

# Loss and KLD are lower-is-better; the rest higher-is-better.
higher_is_better = {
    "Loss": False, "KLD": False, "CC": True, "SIM": True,
    "NSS": True, "AUC": True, "IG": True,
}
metrics = list(higher_is_better.keys())
arrows = {m: ("↑" if higher_is_better[m] else "↓") for m in metrics}

# One fixed colour per model across both figures.
palette = plt.cm.tab10(np.linspace(0, 1, len(df.index)))
colors = dict(zip(df.index, palette))


def normalize_column(col, higher_better, visual_floor=0.06):
    """Min-max a metric column to [0,1] (1.0 = best model, 0.0 = worst).

    visual_floor lifts the worst model off exactly 0 so its bar stays
    visible; pass 0.0 for the heatmap where a true 0 is fine.
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
    width = 0.8 / len(df.index)
    x = np.arange(len(metrics))

    for i, model in enumerate(df.index):
        vals = norm_df.loc[model, metrics].values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=model, color=colors[model])

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


def plot_heatmap(heatmap_metrics=None, out_path="./04_heatmap.png"):
    """Draw the model x metric heatmap. Pass heatmap_metrics to restrict the
    columns (e.g. to drop Loss); defaults to all metrics."""
    hm_metrics = heatmap_metrics if heatmap_metrics is not None else metrics

    norm_df = df.copy()
    for m in hm_metrics:
        norm_df[m] = normalize_column(df[m], higher_is_better[m], visual_floor=0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(norm_df[hm_metrics].values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(hm_metrics)))
    ax.set_xticklabels([f"{m}{arrows[m]}" for m in hm_metrics], fontsize=12)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=12)

    # Cells show the raw score; text goes white on the dark ends of the colormap.
    for i, model in enumerate(df.index):
        for j, m in enumerate(hm_metrics):
            raw_val = df.loc[model, m]
            shade = norm_df.loc[model, m]
            text_color = "white" if shade < 0.3 or shade > 0.85 else "black"
            ax.text(j, i, f"{raw_val:.4f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Normalized performance (green = better)", fontsize=10)
    ax.set_title("Backbone comparison heatmap (SALICON, frozen backbone + shared decoder)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_grouped_bars_normalized()
    plot_heatmap()
    # Loss is a training objective, not a success metric, so also emit a version
    # of the heatmap restricted to the held-out saliency metrics.
    plot_heatmap(heatmap_metrics=[m for m in metrics if m != "Loss"],
                 out_path="./04_heatmap_no_loss.png")
    print("All charts saved to cv_project_saliency/results/")
