# Cross-backbone qualitative comparison for the appendix / defense.
# Lines up several models' saliency predictions on the SAME validation images so
# the reader can SEE why the best backbone wins, not just read Table 1. Default
# columns: our best DINOv3+LoRA vs frozen DINOv3 (roster) vs frozen ResNet-50.
# Uncomment entries in MODELS to add CLIP / DINOv2 / SAM / MAE / supervised ViT.
#
# Like every other script here it only loads trained checkpoints and runs
# inference; nothing trains. Each model is built, run on the handful of selected
# images, scored, then freed -- so even the full 8-model roster fits a 16 GB GPU.
#
# Run on Kaggle:
#   !rm -rf /kaggle/working/cv_project_saliency
#   !git clone https://ghp_...@github.com/peri-7/cv_project_saliency.git
#   !HF_TOKEN="hf_..." python /kaggle/working/cv_project_saliency/lora/visualize_comparison.py
#
# Output (OUT_DIR): cross_backbone_comparison_grid.png
#
# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REPO = '/kaggle/working/cv_project_saliency'

# The first entry (kind='lora') is treated as "ours" and drives image selection.
# Frozen roster checkpoints are raw decoder state_dicts (see notebooks/*.py);
# their tap set is NOT recoverable from the file, so it is stated explicitly.
# best_dinov3_decoder.pth == roster taps [2,5,8,11] (per weight_analysis/dinov3_anal.py).
MODELS = [
    {'label': 'DINOv3 + LoRA (ours)', 'kind': 'lora',
     'ckpt': f'{REPO}/saved_models/best_dinov3_lora_baseline_2tap.pth'},
    {'label': 'DINOv3 (frozen)', 'kind': 'dinov3',
     'ckpt': f'{REPO}/saved_models/best_dinov3_decoder.pth', 'taps': [2, 5, 8, 11]},
    {'label': 'ResNet-50 (frozen)', 'kind': 'resnet',
     'ckpt': f'{REPO}/saved_models/best_resnet_decoder.pth'},

    # --- Optional extra columns (all frozen roster, taps [2,5,8,11]) ---
    # {'label': 'CLIP (frozen)',    'kind': 'clip',   'ckpt': f'{REPO}/saved_models/best_clip_decoder.pth',   'taps': [2, 5, 8, 11]},
    # {'label': 'DINOv2 (frozen)',  'kind': 'dinov2', 'ckpt': f'{REPO}/saved_models/best_dinov2_decoder.pth', 'taps': [2, 5, 8, 11]},
    # {'label': 'SAM (frozen)',     'kind': 'sam',    'ckpt': f'{REPO}/saved_models/best_sam_decoder.pth',    'taps': [2, 5, 8, 11]},
    # {'label': 'MAE (frozen)',     'kind': 'mae',    'ckpt': f'{REPO}/saved_models/best_mae_decoder.pth',    'taps': [2, 5, 8, 11]},
    # {'label': 'ViT sup. (frozen)','kind': 'vit',    'ckpt': f'{REPO}/saved_models/best_vit_decoder.pth',    'taps': [2, 5, 8, 11]},
]

SELECTION  = 'spread'      # 'spread' | 'top' | 'fixed'  (ranked by "ours" CC)
N_EXAMPLES = 6
MAX_SCAN   = 500
FIXED_NAMES = ['COCO_val2014_000000000488.jpg']

CMAP    = 'jet'
ALPHA   = 0.6
OUT_DIR = '/kaggle/working/viz_out'
SEED    = 0

# ---------------------------------------------------------------------------
import os
import sys
import gc

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.append(REPO)

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from src.dataset import LoraDataset
from src.decoder import Decoder, ConvUpDecoder
from src.lora import LoraDinoV3ViT
from src.models import ResNet, DinoV3ViT, DinoV2ViT, ClipViT, SamViT, MaeViT, ViT
from src.losses import CC_Loss, KLD_Loss, SIM_Loss
from src.metrics import nss as nss_metric

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

if 'HF_TOKEN' in os.environ:
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'])

FROZEN_BACKBONES = {'dinov3': DinoV3ViT, 'dinov2': DinoV2ViT, 'clip': ClipViT,
                    'sam': SamViT, 'mae': MaeViT, 'vit': ViT}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

image_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
map_transform = transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()])

base_input_path = '/kaggle/input/datasets/roshan401/salicon'
val_dataset = LoraDataset(
    image_dir=os.path.join(base_input_path, 'images/images/val'),
    maps_dir=os.path.join(base_input_path, 'maps/val'),
    fixations_dir=os.path.join(base_input_path, 'fixations/val'),
    image_transform=image_transform, map_transform=map_transform)
print(f'Validation images: {len(val_dataset)}')

_cc, _kld, _sim = CC_Loss(), KLD_Loss(), SIM_Loss()


# ---------------------------------------------------------------------------
# Model building (each kind knows how to rebuild its architecture + load weights)
# ---------------------------------------------------------------------------
def build_lora(ckpt_path):
    """DINOv3 + LoRA; decoder class and tap count auto-detected from the file."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    dec_state = ckpt['decoder']
    is_upgraded = any(k.startswith(('up_block1', 'fuse_norm', 'refine')) for k in dec_state)
    decoder_cls = ConvUpDecoder if is_upgraded else Decoder
    n_taps = dec_state['channel_compression.weight'].shape[1] // 768
    taps = {2: [10, 11], 4: [2, 5, 8, 11]}[n_taps]
    extractor = LoraDinoV3ViT(tap_blocks=taps).to(device)
    decoder = decoder_cls(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)
    decoder.load_state_dict(dec_state)
    extractor.load_state_dict(ckpt['lora'], strict=False)
    return extractor, decoder


def load_frozen_decoder(decoder, ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'decoder' in state:   # tolerate wrapped saves
        state = state['decoder']
    decoder.load_state_dict(state)


def build_model(entry):
    kind = entry['kind']
    if kind == 'lora':
        extractor, decoder = build_lora(entry['ckpt'])
    elif kind == 'resnet':
        extractor = ResNet().to(device)
        decoder = Decoder(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)
        load_frozen_decoder(decoder, entry['ckpt'])
    else:
        taps = entry.get('taps', [2, 5, 8, 11])
        extractor = FROZEN_BACKBONES[kind](tap_blocks=taps).to(device)
        decoder = Decoder(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)
        load_frozen_decoder(decoder, entry['ckpt'])
    extractor.eval(); decoder.eval()
    return extractor, decoder


@torch.no_grad()
def predict(extractor, decoder, image):
    x = image.unsqueeze(0).to(device)
    logits = decoder(list(extractor(x).values()))
    logits = F.interpolate(logits, size=(480, 640), mode='bilinear', align_corners=False)
    prob = F.softmax(logits.view(1, 1, -1), dim=2).view(1, 1, 480, 640)
    return prob, logits


@torch.no_grad()
def scores(logits, gt_map, fix_map):
    cc = _cc(logits, gt_map.unsqueeze(0).to(device)).item()
    ns = nss_metric(logits, fix_map.unsqueeze(0).to(device))
    return cc, ns


# ---------------------------------------------------------------------------
# Pick images using the FIRST model (ours), then keep it for the render pass.
# ---------------------------------------------------------------------------
def select_with(extractor, decoder):
    names = val_dataset.image_filenames
    if SELECTION == 'fixed':
        idxs = [names.index(n) for n in FIXED_NAMES]
        return idxs, ['requested'] * len(idxs)

    n_scan = min(MAX_SCAN, len(val_dataset))
    loader = DataLoader(Subset(val_dataset, range(n_scan)), batch_size=8,
                        shuffle=False, num_workers=2)
    ccs = []
    print(f'Scoring {n_scan} val images with "{MODELS[0]["label"]}"...')
    with torch.no_grad():
        for images, gt, _ in loader:
            images = images.to(device)
            logits = decoder(list(extractor(images).values()))
            logits = F.interpolate(logits, size=(480, 640), mode='bilinear', align_corners=False)
            b = images.shape[0]
            pf = logits.view(b, -1); gf = gt.view(b, -1).to(device)
            pc = pf - pf.mean(1, keepdim=True); gc = gf - gf.mean(1, keepdim=True)
            cc = (pc * gc).sum(1) / (pc.norm(dim=1) * gc.norm(dim=1) + 1e-7)
            ccs.extend(cc.cpu().tolist())
    order = np.argsort(ccs)[::-1]
    if SELECTION == 'top':
        return list(order[:N_EXAMPLES]), ['top'] * N_EXAMPLES
    fracs = np.linspace(0.02, 0.85, N_EXAMPLES)
    picks = (fracs * (len(order) - 1)).astype(int)
    return list(order[picks]), [f'CC pct {100 - int(f * 100)}' for f in fracs]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def denorm(image):
    return (image * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def norm01(a):
    a = a - a.min(); m = a.max()
    return a / m if m > 0 else a


def overlay(rgb, sal01):
    heat = plt.get_cmap(CMAP)(sal01)[..., :3]
    a = (ALPHA * sal01)[..., None]
    return (1 - a) * rgb + a * heat


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = val_dataset.image_filenames

    # 1) Build "ours", select images, cache its predictions, then free it.
    extractor, decoder = build_model(MODELS[0])
    idxs, tags = select_with(extractor, decoder)

    cache = []   # per selected image: display + ground truth + fixed inputs
    for idx in idxs:
        image, gt_map, fix_map = val_dataset[idx]
        cache.append({'idx': int(idx), 'image': image, 'gt': gt_map, 'fix': fix_map,
                      'rgb': denorm(image), 'gt01': norm01(gt_map[0].cpu().numpy())})

    preds = {}   # label -> list (aligned with idxs) of (pred01, cc, nss)
    row = []
    for c in cache:
        prob, logits = predict(extractor, decoder, c['image'])
        cc, ns = scores(logits, c['gt'], c['fix'])
        row.append((norm01(prob[0, 0].cpu().numpy()), cc, ns))
    preds[MODELS[0]['label']] = row
    del extractor, decoder; gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 2) Every other model: build, predict on the selected images, score, free.
    for entry in MODELS[1:]:
        print(f'Running {entry["label"]}...')
        extractor, decoder = build_model(entry)
        row = []
        for c in cache:
            prob, logits = predict(extractor, decoder, c['image'])
            cc, ns = scores(logits, c['gt'], c['fix'])
            row.append((norm01(prob[0, 0].cpu().numpy()), cc, ns))
        preds[entry['label']] = row
        del extractor, decoder; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 3) Render: Input | Ground truth | <one column per model>.
    labels = [m['label'] for m in MODELS]
    col_titles = ['Input image', 'Ground truth'] + labels
    ncols = len(col_titles)
    fig = plt.figure(figsize=(ncols * 2.7, len(cache) * 2.3 + 0.6))
    gs = gridspec.GridSpec(len(cache), ncols, figure=fig, wspace=0.04, hspace=0.10)

    for r, c in enumerate(cache):
        row_imgs = [c['rgb'], overlay(c['rgb'], c['gt01'])]
        row_metrics = [None, None]
        for lb in labels:
            pred01, cc, ns = preds[lb][r]
            row_imgs.append(overlay(c['rgb'], pred01))
            row_metrics.append((cc, ns))
        for col in range(ncols):
            ax = fig.add_subplot(gs[r, col])
            ax.imshow(row_imgs[col]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[col], fontsize=10, pad=6)
            if col == 0:
                ax.set_ylabel(f'{names[c["idx"]]}\n{tags[r]}', fontsize=7, rotation=0,
                              ha='right', va='center', labelpad=6, color='0.35')
            if row_metrics[col] is not None:
                cc, ns = row_metrics[col]
                ax.text(0.5, -0.045, f'CC {cc:.2f}  NSS {ns:.2f}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=8,
                        color='0.15' if col == 2 else '0.35',
                        fontweight='bold' if col == 2 else 'normal')

    fig.suptitle('Saliency predictions across backbones — SALICON validation',
                 fontsize=13, y=0.995)
    path = os.path.join(OUT_DIR, 'cross_backbone_comparison_grid.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved figure: {path}')
    # Sanity print: mean CC per model over the shown images (a wrong tap set
    # would show up here as an implausibly low number).
    for lb in labels:
        mcc = np.mean([cc for _, cc, _ in preds[lb]])
        mns = np.mean([ns for _, _, ns in preds[lb]])
        print(f'  mean over shown ({lb}): CC {mcc:.3f}  NSS {mns:.3f}')


if __name__ == '__main__':
    main()
    print('Done.')
