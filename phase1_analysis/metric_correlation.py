"""
metric_correlation.py
---------------------
Metric-metric agreement analysis for Phase I.

Idea: each metric induces a *ranking* of the backbones. If two metrics rank the
models the same way, they are redundant (they "see" the same thing). We quantify
this with Spearman rank correlation between every pair of metrics, computed over
the models, and visualise it as a heatmap. Clusters of highly-correlated metrics
are exactly the "Similarity cluster" idea from Bylinskii et al. [s6] -- but here
documented on OUR OWN data instead of borrowed.

Only needs numpy + matplotlib (no torch / no data / no GPU), because the metric
scores are the ones already reported in results/plot_results.py.

Run:
    python phase1_analysis/metric_correlation.py
Outputs:
    phase1_analysis/figures/metric_correlation_heatmap.png
    phase1_analysis/figures/metric_correlation_heatmap_no_resnet.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

# Phase I scores (same numbers as results/plot_results.py).
# IG uses the empirical center-bias baseline, consistent with the methodology.
MODELS = ["ResNet", "ViT", "CLIP", "MAE", "DiNOv2", "DiNOv3", "SAM"]
SCORES = {
    "Loss": [1.5806, 1.0398, 0.6037, 0.8191, 0.5515, 0.5529, 0.7691],
    "KLD":  [0.2995, 0.2493, 0.2099, 0.2289, 0.2050, 0.2064, 0.2245],
    "CC":   [0.6824, 0.6951, 0.7156, 0.7025, 0.7121, 0.7257, 0.7051],
    "SIM":  [0.7321, 0.7579, 0.7794, 0.7676, 0.7862, 0.7853, 0.7707],
    "NSS":  [1.2975, 1.3117, 1.3519, 1.3200, 1.3447, 1.3712, 1.3255],
    "AUC":  [0.8555, 0.8638, 0.8686, 0.8668, 0.8694, 0.8697, 0.8677],
    "IG":   [0.7906, 0.8538, 0.9267, 0.8894, 0.9356, 0.9366, 0.8990],
}
# Direction: True = higher is better. We flip lower-is-better metrics so that a
# POSITIVE correlation always means "the two metrics agree on which model is better".
HIGHER_IS_BETTER = {
    "Loss": False, "KLD": False, "CC": True, "SIM": True,
    "NSS": True, "AUC": True, "IG": True,
}
METRICS = list(SCORES.keys())


def _rankdata(a):
    """Average ranks (ties -> mean rank), numpy-only replacement for scipy."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # resolve ties by averaging
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return ranks


def spearman_matrix(matrix):
    """matrix: [n_metrics, n_models] already oriented higher-is-better."""
    ranks = np.vstack([_rankdata(row) for row in matrix])
    # Spearman = Pearson on ranks
    return np.corrcoef(ranks)


def seriation_order(corr):
    """Order metrics so similar ones sit together (1-D embedding via top eigvec)."""
    # use the leading eigenvector of the correlation matrix as a 1-D coordinate
    w, v = np.linalg.eigh(corr)
    coord = v[:, -1]
    return np.argsort(coord)


def build_matrix(models_subset):
    idx = [MODELS.index(m) for m in models_subset]
    rows = []
    for m in METRICS:
        vals = np.array(SCORES[m])[idx]
        if not HIGHER_IS_BETTER[m]:
            vals = -vals  # orient so higher = better
        rows.append(vals)
    return np.array(rows)


def plot_heatmap(models_subset, fname, title):
    mat = build_matrix(models_subset)
    corr = spearman_matrix(mat)
    order = seriation_order(corr)
    corr_o = corr[np.ix_(order, order)]
    labels = [METRICS[i] for i in order]

    # All correlations are positive & high (tiny N), so stretch the colour range
    # to the observed off-diagonal minimum to make the cluster blocks visible.
    off = corr_o[~np.eye(len(labels), dtype=bool)]
    vmin = np.floor(off.min() * 20) / 20  # round down to nearest 0.05
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_o, cmap="YlOrRd", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr_o[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr_o[i, j] > (vmin + 1) / 2 else "black",
                    fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Spearman rank correlation (oriented higher=better, scale [{vmin:.2f}, 1.0])")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUTDIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")
    print("order:", labels)
    return corr_o, labels


if __name__ == "__main__":
    print("=== all 7 models (incl. ResNet baseline) ===")
    plot_heatmap(MODELS,
                 "metric_correlation_heatmap.png",
                 "Metric-metric agreement (Spearman) - all models")

    print("\n=== 6 backbones (ResNet excluded, matches report definition) ===")
    plot_heatmap([m for m in MODELS if m != "ResNet"],
                 "metric_correlation_heatmap_no_resnet.png",
                 "Metric-metric agreement (Spearman) - backbones only")
