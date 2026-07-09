# Qualitative-visualization script for the paper's best model.
# Produces the "best-model predictions" figure asked for in the supplementary /
# appendix: a grid of  Input | Ground-truth | Prediction | Error  over a set of
# SALICON validation images, each panel annotated with the paper's own metrics
# (CC / NSS / SIM / KLD). Nothing here trains — it only loads a trained
# checkpoint and runs inference, exactly like lora/dino3_lora_2tap_eval.py.
#
# Run on Kaggle (same environment as every other script in this repo):
#   !rm -rf /kaggle/working/cv_project_saliency
#   !git clone https://ghp_...@github.com/peri-7/cv_project_saliency.git
#   !HF_TOKEN="hf_..." python /kaggle/working/cv_project_saliency/lora/visualize_predictions.py
#
# Outputs (written to OUT_DIR):
#   best_model_predictions_grid.png   -- the combined appendix figure
#   panel_<rank>_<name>.png           -- one high-res panel per example (for slides)
#
# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Point this at your best checkpoint. The script AUTO-DETECTS the decoder type
# (lightweight `Decoder` vs progressive `ConvUpDecoder`) and the number of taps
# from the checkpoint itself, so any of the LoRA checkpoints below just works:
#
#   best_dinov3_lora_upgraded_2tap.pth  -> ConvUpDecoder, taps [10,11]   (committed)
#   best_dinov3_lora_simpledecoder.pth  -> Decoder,       taps [2,5,8,11] (committed)
#   best_dinov3_lora_upgraded.pth       -> ConvUpDecoder, taps [2,5,8,11] (committed)
#   best_dinov3_lora_baseline_2tap.pth  -> Decoder,       taps [10,11]    (the PAPER-BEST
#                                          config; produced on Kaggle, not in the repo --
#                                          mount it as a Kaggle input and point CKPT_PATH here)
CKPT_PATH = '/kaggle/working/cv_project_saliency/saved_models/best_dinov3_lora_baseline_2tap.pth'

# Leave as None to infer taps from the checkpoint (2 -> [10,11], 4 -> [2,5,8,11]).
# Set explicitly only if you tapped non-standard blocks.
TAP_BLOCKS = None

# How the example images are chosen from the validation set:
#   'spread' -> a spread across the CC distribution (mostly strong, one hard case);
#               the honest "here is what the model does" choice.  [default]
#   'top'    -> the highest-CC images (best-case showcase).
#   'fixed'  -> the exact filenames listed in FIXED_NAMES.
SELECTION   = 'spread'
N_EXAMPLES  = 6
MAX_SCAN    = 500          # cap on how many val images to score for selection (speed)
FIXED_NAMES = ['COCO_val2014_000000000488.jpg']   # only used when SELECTION == 'fixed'

CMAP    = 'jet'            # saliency-field convention; 'turbo'/'inferno' are
                           # perceptually better if you prefer a modern look.
ALPHA   = 0.6             # overlay strength of the heatmap over the image
OUT_DIR = '/kaggle/working/viz_out'
SEED    = 0

# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.append('/kaggle/working/cv_project_saliency/')

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import LoraDataset
from src.decoder import Decoder, ConvUpDecoder
from src.lora import LoraDinoV3ViT
from src.losses import KLD_Loss, CC_Loss, SIM_Loss
from src.metrics import nss as nss_metric

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# HF login is only needed to pull the gated DINOv3 backbone weights.
if 'HF_TOKEN' in os.environ:
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'])

# ImageNet normalization used everywhere in the study (see the training scripts).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

image_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
map_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

base_input_path = '/kaggle/input/datasets/roshan401/salicon'
val_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, 'images/images/val'),
    maps_dir=os.path.join(base_input_path, 'maps/val'),
    fixations_dir=os.path.join(base_input_path, 'fixations/val'),
    image_transform=image_transform,
    map_transform=map_transform,
)
print(f'Validation images: {len(val_dataset)}')


# ---------------------------------------------------------------------------
# Rebuild the exact architecture from the checkpoint, then load the weights.
# ---------------------------------------------------------------------------
def build_from_checkpoint(ckpt_path):
    """Infer decoder class + tap count from the saved decoder state, so the same
    script serves every LoRA checkpoint without hand-editing flags."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    dec_state = ckpt['decoder']

    # Decoder vs ConvUpDecoder: only the progressive head has 'up_block1'/'fuse_norm'.
    is_upgraded = any(k.startswith(('up_block1', 'fuse_norm', 'refine')) for k in dec_state)
    decoder_cls = ConvUpDecoder if is_upgraded else Decoder

    # channel_compression.weight is [hidden, 768*n_taps, 1, 1] -> recover n_taps.
    in_ch = dec_state['channel_compression.weight'].shape[1]
    n_taps = in_ch // 768

    taps = TAP_BLOCKS
    if taps is None:
        taps = {2: [10, 11], 4: [2, 5, 8, 11]}.get(n_taps)
        if taps is None:
            raise ValueError(
                f'Cannot infer tap blocks for {n_taps} taps; set TAP_BLOCKS explicitly.')
    assert len(taps) == n_taps, (
        f'TAP_BLOCKS={taps} has {len(taps)} taps but the checkpoint decoder '
        f'expects {n_taps}.')

    print(f'Detected: decoder={decoder_cls.__name__}, taps={taps}')

    extractor = LoraDinoV3ViT(tap_blocks=taps).to(device)
    decoder = decoder_cls(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)

    decoder.load_state_dict(dec_state)
    if 'lora' in ckpt:
        extractor.load_state_dict(ckpt['lora'], strict=False)
        print('LoRA adapters loaded.')
    extractor.eval()
    decoder.eval()
    return extractor, decoder


extractor, decoder = build_from_checkpoint(CKPT_PATH)

# Per-image metric helpers (batch-of-1 -> the paper's exact definitions).
_kld, _cc, _sim = KLD_Loss(), CC_Loss(), SIM_Loss()


@torch.no_grad()
def predict(image):
    """image: [3,H,W] normalized -> (prob_map[H,W], logits[1,1,H,W]) at 480x640."""
    x = image.unsqueeze(0).to(device)
    feats = list(extractor(x).values())
    logits = decoder(feats)
    logits = F.interpolate(logits, size=(480, 640), mode='bilinear', align_corners=False)
    prob = F.softmax(logits.view(1, 1, -1), dim=2).view(1, 1, 480, 640)
    return prob, logits


@torch.no_grad()
def metrics_for(logits, prob, gt_map, fix_map):
    gt = gt_map.unsqueeze(0).to(device)
    fx = fix_map.unsqueeze(0).to(device)
    return {
        'CC': _cc(logits, gt).item(),
        'NSS': nss_metric(logits, fx),
        'SIM': _sim(prob, gt).item(),
        'KLD': _kld(prob, gt).item(),
    }


# ---------------------------------------------------------------------------
# Choose which validation images to display.
# ---------------------------------------------------------------------------
def select_indices():
    names = val_dataset.image_filenames

    if SELECTION == 'fixed':
        idxs = [names.index(n) for n in FIXED_NAMES]
        return idxs, [f'requested' for _ in idxs]

    # Score up to MAX_SCAN images by CC and rank them.
    n_scan = min(MAX_SCAN, len(val_dataset))
    loader = DataLoader(torch.utils.data.Subset(val_dataset, range(n_scan)),
                        batch_size=8, shuffle=False, num_workers=2)
    ccs = []
    print(f'Scoring {n_scan} val images to pick {N_EXAMPLES} examples...')
    with torch.no_grad():
        for images, gt, _ in loader:
            images = images.to(device)
            feats = list(extractor(images).values())
            logits = decoder(feats)
            logits = F.interpolate(logits, size=(480, 640), mode='bilinear', align_corners=False)
            b = images.shape[0]
            pf = logits.view(b, -1)
            gf = gt.view(b, -1).to(device)
            pc = pf - pf.mean(1, keepdim=True)
            gc = gf - gf.mean(1, keepdim=True)
            cc = (pc * gc).sum(1) / (pc.norm(dim=1) * gc.norm(dim=1) + 1e-7)
            ccs.extend(cc.cpu().tolist())
    order = np.argsort(ccs)[::-1]          # best CC first

    if SELECTION == 'top':
        chosen = order[:N_EXAMPLES]
        tags = ['top' for _ in chosen]
    else:  # 'spread' -> mostly strong, ending on a hard case (honest range)
        fracs = np.linspace(0.02, 0.85, N_EXAMPLES)
        picks = (fracs * (len(order) - 1)).astype(int)
        chosen = order[picks]
        tags = [f'CC pct {100 - int(f * 100)}' for f in fracs]
    return list(chosen), tags


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def denorm(image):
    """Normalized [3,H,W] tensor -> HWC uint8-ish float in [0,1] for display."""
    x = image * IMAGENET_STD + IMAGENET_MEAN
    return x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def norm01(a):
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


def overlay(rgb, sal01, cmap=CMAP, alpha=ALPHA):
    """Alpha-blend a saliency heatmap onto the RGB image, alpha scaled by
    saliency so quiet regions keep the photo and hot regions show the map."""
    heat = plt.get_cmap(cmap)(sal01)[..., :3]
    a = (alpha * sal01)[..., None]
    return (1 - a) * rgb + a * heat


def render(indices, tags):
    os.makedirs(OUT_DIR, exist_ok=True)
    names = val_dataset.image_filenames
    ncols = 4  # Input | Ground truth | Prediction | Error
    col_titles = ['Input image', 'Ground truth', 'Prediction (ours)', 'Abs. error']

    fig = plt.figure(figsize=(ncols * 3.1, len(indices) * 2.5 + 0.6))
    gs = gridspec.GridSpec(len(indices), ncols, figure=fig,
                           wspace=0.04, hspace=0.10)

    for r, (idx, tag) in enumerate(zip(indices, tags)):
        image, gt_map, fix_map = val_dataset[idx]
        prob, logits = predict(image)
        m = metrics_for(logits, prob, gt_map, fix_map)

        rgb = denorm(image)
        gt = norm01(gt_map[0].cpu().numpy())
        pred = norm01(prob[0, 0].cpu().numpy())
        err = np.abs(pred - gt)

        panels = [
            (rgb, None),
            (overlay(rgb, gt), None),
            (overlay(rgb, pred), None),
            (err, 'magma'),
        ]
        for c, (img, cm) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img, cmap=cm, vmin=0, vmax=1) if cm else ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(f'{names[idx]}\n{tag}', fontsize=7, rotation=0,
                              ha='right', va='center', labelpad=6, color='0.35')
            if c == 2:
                ax.text(0.5, -0.06,
                        f"CC {m['CC']:.2f}  NSS {m['NSS']:.2f}  "
                        f"SIM {m['SIM']:.2f}  KLD {m['KLD']:.2f}",
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=8, color='0.2')

        # Also drop a standalone high-res panel for slides.
        pf = plt.figure(figsize=(ncols * 2.6, 2.4))
        for c, (img, cm) in enumerate(panels):
            a = pf.add_subplot(1, ncols, c + 1)
            a.imshow(img, cmap=cm, vmin=0, vmax=1) if cm else a.imshow(img)
            a.set_xticks([]); a.set_yticks([]); a.set_title(col_titles[c], fontsize=10)
        pf.suptitle(f"{names[idx]}   |   CC {m['CC']:.2f}  NSS {m['NSS']:.2f}  "
                    f"SIM {m['SIM']:.2f}  KLD {m['KLD']:.2f}", fontsize=9, y=1.02)
        pf.tight_layout()
        pf.savefig(os.path.join(OUT_DIR, f'panel_{r:02d}_{os.path.splitext(names[idx])[0]}.png'),
                   dpi=200, bbox_inches='tight')
        plt.close(pf)

        print(f"  [{r}] {names[idx]}  CC={m['CC']:.3f} NSS={m['NSS']:.3f} "
              f"SIM={m['SIM']:.3f} KLD={m['KLD']:.3f}")

    fig.suptitle('DINOv3 + LoRA (best model) — SALICON validation predictions',
                 fontsize=13, y=0.995)
    grid_path = os.path.join(OUT_DIR, 'best_model_predictions_grid.png')
    fig.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved figure: {grid_path}')
    print(f'Saved {len(indices)} standalone panels in: {OUT_DIR}')


if __name__ == '__main__':
    idxs, tags = select_indices()
    render(idxs, tags)
    print('Done.')
