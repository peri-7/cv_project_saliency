"""
Shared logic for the decoder interpretability study (see weight_analysis/PLAN.md).

Analysis A — tap_importance(): per-tap importance matrix M[i, k] built from the
    decoder's channel_compression weights, scale-corrected by per-channel feature
    stds measured on the val set.
Analysis B — ablation_study(): leave-one-tap-out ablation, reusing
    src.training_online.test_model_online so the 7-tuple is computed by exactly
    the same code as the real benchmarks.

run_analysis() orchestrates both and saves figures/CSVs; the per-model scripts
(weight_analysis/resnet_anal.py etc.) only build the backbone + decoder and call it.
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')  # headless (Kaggle scripts)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training_online import test_model_online

METRIC_NAMES = ['loss', 'kld', 'cc', 'sim', 'nss', 'auc', 'ig']


# ---------------------------------------------------------------------------
# Analysis A — per-tap importance
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_channel_stds(extractor, dataloader, device, max_batches=None):
    """
    Per-channel std of the concatenated feature tensor over the val set.

    The stds are measured AFTER replicating the decoder's upsample-to-finest-grid
    step, because that is the tensor channel_compression actually multiplies.
    (For the ViT family every tap shares one grid, so this is a no-op; for ResNet
    the coarse stages get bilinearly upsampled, which slightly smooths them, and
    we want the std of what the conv sees.)

    Returns a CPU float tensor [C_total] ordered tap1..tapK, matching the channel
    layout of channel_compression.weight.
    """
    extractor.eval()
    sums, sq_sums = None, None
    count = 0

    for batch_idx, (images, _, _) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        feats = list(extractor(images).values())

        # Mirror Decoder.forward: upsample every tap to the finest (first) grid.
        target_size = feats[0].shape[2:]
        b = feats[0].shape[0]
        h, w = target_size

        if sums is None:
            num_ch = sum(f.shape[1] for f in feats)
            sums = torch.zeros(num_ch, dtype=torch.float64, device=device)
            sq_sums = torch.zeros(num_ch, dtype=torch.float64, device=device)

        offset = 0
        for f in feats:
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear',
                                  align_corners=False)
            c = f.shape[1]
            sums[offset:offset + c] += f.sum(dim=(0, 2, 3)).double()
            sq_sums[offset:offset + c] += f.pow(2).sum(dim=(0, 2, 3)).double()
            offset += c
        count += b * h * w

    mean = sums / count
    var = sq_sums / count - mean ** 2
    return var.clamp_min_(0).sqrt().float().cpu()


def tap_importance(decoder, channel_stds, out_channels):
    """
    M[i, k] = sum over channels c in tap k of |W[i, c]| * std(x_c).

    Tap boundaries come from out_channels (e.g. ResNet's [256, 512, 1024, 2048]),
    never hard-coded. Returns a CPU tensor [hidden_dim, num_taps].
    """
    W = decoder.channel_compression.weight.detach().cpu().flatten(1)  # [hidden, C_total]
    assert W.shape[1] == sum(out_channels) == channel_stds.shape[0]

    contrib = W.abs() * channel_stds.unsqueeze(0)
    per_tap = torch.split(contrib, list(out_channels), dim=1)
    return torch.stack([chunk.sum(dim=1) for chunk in per_tap], dim=1)


def plot_tap_importance(M, tap_labels, save_path, title):
    """
    Two-panel figure:
      left  — heatmap of M, rows row-normalized (so banding is readable) and
              sorted by dominant tap, then by dominance strength within group;
      right — raw column sums (the headline: which tap the decoder relies on).
    """
    M = M.detach().cpu()
    row_norm = M / M.sum(dim=1, keepdim=True).clamp_min(1e-12)

    dominant = row_norm.argmax(dim=1)
    strength = row_norm.max(dim=1).values
    order = sorted(range(M.shape[0]),
                   key=lambda i: (dominant[i].item(), -strength[i].item()))

    col_sums = M.sum(dim=0)
    col_frac = col_sums / col_sums.sum()

    fig, (ax_hm, ax_bar) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [2, 1]})

    im = ax_hm.imshow(row_norm[order].numpy(), aspect='auto',
                      cmap='viridis', vmin=0, vmax=1)
    ax_hm.set_xticks(range(len(tap_labels)))
    ax_hm.set_xticklabels(tap_labels, rotation=20)
    ax_hm.set_ylabel(f'output neuron (sorted by dominant tap, n={M.shape[0]})')
    ax_hm.set_title('per-neuron tap importance (row-normalized)')
    fig.colorbar(im, ax=ax_hm, fraction=0.046)

    ax_bar.bar(range(len(tap_labels)), col_sums.numpy())
    ax_bar.set_xticks(range(len(tap_labels)))
    ax_bar.set_xticklabels(tap_labels, rotation=20)
    ax_bar.set_title('total tap importance (column sums)')
    for k, (s, fr) in enumerate(zip(col_sums, col_frac)):
        ax_bar.text(k, s.item(), f'{fr.item():.1%}', ha='center', va='bottom')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis B — leave-one-tap-out ablation
# ---------------------------------------------------------------------------

class TapAblatedExtractor(nn.Module):
    """
    Wraps a frozen backbone and zeros out one tap's feature tensor, so the
    unmodified test_model_online can be reused for the ablation passes.
    """

    def __init__(self, extractor, zero_tap):
        super().__init__()
        self.extractor = extractor
        self.zero_tap = zero_tap
        self.out_channels = extractor.out_channels

    def forward(self, x):
        feats = self.extractor(x)
        return {k: torch.zeros_like(v) if i == self.zero_tap else v
                for i, (k, v) in enumerate(feats.items())}


def ablation_study(extractor, decoder, dataloader, criterion, device, tap_labels):
    """
    Baseline pass + one pass per zeroed tap, each through test_model_online.
    Returns an ordered dict-like list of (row_label, 7-tuple).
    """
    results = []

    print('  ablation: baseline pass...')
    baseline = test_model_online(extractor, decoder, dataloader, criterion, device)
    results.append(('baseline', baseline))

    for k, label in enumerate(tap_labels):
        print(f'  ablation: zeroing {label}...')
        ablated = TapAblatedExtractor(extractor, zero_tap=k)
        metrics = test_model_online(ablated, decoder, dataloader, criterion, device)
        results.append((f'zero {label}', metrics))

    return results


def print_ablation_table(results):
    baseline = results[0][1]
    header = f"{'':<18}" + ''.join(f'{name:>12}' for name in METRIC_NAMES)
    print(header)
    for row_label, metrics in results:
        line = f'{row_label:<18}' + ''.join(f'{v:>12.4f}' for v in metrics)
        print(line)
        if row_label != 'baseline':
            deltas = [v - b for v, b in zip(metrics, baseline)]
            print(f"{'  (delta)':<18}" + ''.join(f'{d:>+12.4f}' for d in deltas))


def save_ablation_csv(results, save_path):
    baseline = results[0][1]
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row'] + METRIC_NAMES
                        + [f'delta_{n}' for n in METRIC_NAMES])
        for row_label, metrics in results:
            deltas = [v - b for v, b in zip(metrics, baseline)]
            writer.writerow([row_label] + [f'{v:.6f}' for v in metrics]
                            + [f'{d:.6f}' for d in deltas])


def save_importance_csv(M, tap_labels, save_path):
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['neuron'] + list(tap_labels))
        for i, row in enumerate(M.tolist()):
            writer.writerow([i] + [f'{v:.6f}' for v in row])
        writer.writerow(['column_sum'] + [f'{v:.6f}' for v in M.sum(dim=0).tolist()])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_analysis(extractor, decoder, val_loader, criterion, device,
                 tap_labels, model_name, out_dir, max_std_batches=None):
    """
    Runs Analysis A then Analysis B and saves all artifacts to out_dir:
      {model_name}_tap_importance.png / .csv   (Analysis A)
      {model_name}_ablation.csv                (Analysis B)
    Everything is inference-only; nothing trains.
    """
    os.makedirs(out_dir, exist_ok=True)
    extractor.eval()
    decoder.eval()

    print(f'[{model_name}] Analysis A: collecting per-channel stds on val...')
    stds = collect_channel_stds(extractor, val_loader, device,
                                max_batches=max_std_batches)

    M = tap_importance(decoder, stds, extractor.out_channels)
    col_sums = M.sum(dim=0)
    col_frac = col_sums / col_sums.sum()
    print(f'[{model_name}] tap importance (column sums):')
    for label, s, fr in zip(tap_labels, col_sums, col_frac):
        print(f'    {label:<20} {s.item():10.4f}  ({fr.item():.1%})')

    heatmap_path = os.path.join(out_dir, f'{model_name}_tap_importance.png')
    plot_tap_importance(M, tap_labels, heatmap_path,
                        title=f'{model_name}: decoder tap importance')
    save_importance_csv(M, tap_labels,
                        os.path.join(out_dir, f'{model_name}_tap_importance.csv'))
    print(f'[{model_name}] saved {heatmap_path}')

    print(f'[{model_name}] Analysis B: leave-one-tap-out ablation on val...')
    results = ablation_study(extractor, decoder, val_loader, criterion,
                             device, tap_labels)
    print_ablation_table(results)
    save_ablation_csv(results, os.path.join(out_dir, f'{model_name}_ablation.csv'))
    print(f'[{model_name}] done.')

    return M, results
