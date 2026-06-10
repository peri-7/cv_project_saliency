# cv_project_saliency

A comparison study of **frozen pretrained backbones** for visual saliency
prediction on the [SALICON](http://salicon.net/) dataset. Every backbone is frozen
and feeds a single **shared, trainable decoder** — only the decoder ever trains, so
any difference in score reflects the **backbone's features**, not the head. The
question being asked is: *which pretraining objective yields features most useful
for saliency?*

## Backbones

The study targets a roster of **six ViT-B-scale backbones**, deliberately chosen to
have approximately the same parameter count (~86–90M) so the *pretraining objective*
— not model capacity — is the variable. ResNet-50 is the original baseline and stays
in for reference (it is not ViT-B, so it sits outside the param-matched family).

| Backbone      | Pretraining objective        | Status                              |
|---------------|------------------------------|-------------------------------------|
| ResNet-50     | ImageNet supervised          | ✅ implemented — `ResNet` (baseline) |
| ViT-B         | ImageNet supervised          | ✅ implemented — `ViT`    (baseline) |
| SAM ViT-B     | Segment Anything (SA-1B)     | ✅ implemented — `SamViT`            |
| DINOv2 ViT-B  | self-distillation            | ✅ implemented — `DinoV2ViT`         |
| DINOv3 ViT-B  | self-supervised (Gram-anchor)| ✅ implemented — `DinoV3ViT`         |
| MAE ViT-B     | masked autoencoding          | ✅ implemented — `MaeViT`            |
| CLIP ViT-B    | image–text contrastive       | ✅ implemented — `ClipViT`           |

Every backbone follows one contract so the shared decoder consumes it unchanged:
**freeze all params, lock `eval()`, return a dict of multi-scale feature tensors, and
expose an `out_channels` list.** ResNet draws its four scales from genuinely different
resolutions (conv stages /4 /8 /16 /32). The plain-ViT family has only one resolution,
so the comparable multi-scale signal comes from tapping **four evenly-spaced transformer
blocks** (blocks 2/5/8/11 for ViT-B's 12) — variety in abstraction *depth*, not spatial
resolution. This 4-block tap is the shared contract across the whole ViT family, so
fairness is structural rather than a per-model tuning choice.

> **SAM is the plumbing odd-one-out.** Its timm encoder uses windowed attention and
> keeps features as spatial `[B, H, W, C]` with no CLS token, so `SamViT` just permutes
> to channels-first. The rest of the ViT-Bs are token-sequence models (`[B, 1+N, C]`)
> that need a token→grid reshape plus prefix-token drop, so those wrappers are
> near-copies of each other over a shared pattern. Two carry extra plumbing:
> **DINOv2 is patch-14**, so 480×640 isn't divisible by 14 — the input is cropped to
> 476×630 (a clean 34×45 grid) before patch embedding. **DINOv3 is patch-16** (so no
> crop, clean 30×40 grid) but inserts **register tokens** between CLS and the patch
> tokens, so instead of dropping one leading token it keeps the *last* `H_p·W_p` tokens.
> The supervised ViT, MAE, and CLIP are plain patch-16 with a single CLS token.

## Approach

Each model is a **frozen backbone + a small trainable decoder**:

- The backbone is frozen and run in `eval()` mode. It emits multi-scale feature maps
  from several depths (low-level contrast → high-level semantics).
- The decoder fuses those features and predicts a single-channel saliency map. It is
  the only part that trains, so any score difference between runs reflects the
  **backbone**, not the head.

For the comparison to be valid the backbone must be the *only* thing that changes, so
everything else is held fixed across runs: decoder `hidden_dim`, optimizer and learning
rate (`Adam`, `lr=1e-4`), epoch count / LR schedule, batch size, input resolution
(480×640), the ImageNet input normalization, and the train/val split.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/models.py` | Frozen backbones: `ResNet`, `SamViT`, `MaeViT`, `ClipViT`, `DinoV2ViT`, `DinoV3ViT`, `ViT`, multi-scale feature extractors |
| `src/decoder.py` | Shared trainable decoder head (`GroupNorm`, for small batch sizes) |
| `src/losses.py` | `Composite_Loss` = `10·KLD − CC − SIM` (KLD/SIM on softmax probs, CC on logits) |
| `src/metrics.py` | Discrete test-time metrics: NSS, AUC-Judd, Information Gain |
| `src/dataset.py` | Datasets for raw images, cached features, and end-to-end training |
| `src/training_online.py` | End-to-end train / eval / test loop (the primary route) |
| `src/training.py` | Offline two-phase loop (cache features to disk, then train decoder) |
| `local_tests/` | Local smoke tests (run against `mini_data/`): `testing2`/`_sam`/`_mae`/`_vit` (online), `testing` (offline) |
| `kaggle_tests/` | Kaggle scripts (clone repo, Kaggle data paths): `{clip,mae,vit,dinov2,dinov3}_smoke_test.py` + the decoder-width sweep (`sweep.py` / `new-sweep.ipynb`, used to lock `hidden_dim` = 256) |
| `notebooks/` | Kaggle training mirrors (`resnet`, `sam`, `mae`, `clip`, `vit`, `dinov2`, `dinov3`) + `.ipynb` real runs |
| `inspect_mat.py`, `mat_to_png.py` | Utilities for SALICON `.mat` fixation files |

The final benchmark reports the full metric set: **KLD, CC, SIM** (continuous, from the
loss) and **NSS, AUC-Judd, Information Gain** (discrete, from `metrics.py`).

## Data

Expects SALICON-style data, standardized to **480×640**:

```
data/
  images/{train,val,test}/      # RGB stimuli (.jpg/.png)
  maps/{train,val,test}/        # continuous saliency maps (.png)  -> used by the loss
  fixations/{train,val,test}/   # discrete fixations (.mat)        -> used by the metrics
```

The `test` split has no maps, so final metrics are reported on `val` as a proxy.
SALICON `.mat` files store gaze as a `(num_subjects, 1)` struct; the loaders union the
fixations of **all** subjects to build the ground-truth fixation map.

## Setup & usage

Run from the repo root so `src/` is importable. The ViT backbones load through
[`timm`](https://github.com/huggingface/pytorch-image-models) (`pip install timm`;
preinstalled on Kaggle); ResNet only needs `torchvision`.

```bash
# Online end-to-end smoke tests (backbone runs live, decoder trains)
# Run as modules from the repo root so `src/` is importable.
python -m local_tests.testing2     # ResNet baseline
python -m local_tests.testing_sam  # SAM ViT-B
python -m local_tests.testing_mae  # MAE ViT-B
python -m local_tests.testing_vit  # supervised ViT-B

# Offline two-phase smoke test: extract features to disk, then train the decoder
python -m local_tests.testing

# Inspect the struct of a SALICON .mat fixation file
python inspect_mat.py

# Convert a .mat fixation file to a binary PNG fixation map
python mat_to_png.py <input.mat> [output.png] [--radius N]
```

Real training runs use the Kaggle notebooks in `notebooks/` (online route), which point
at a Kaggle SALICON dataset path and save the best decoder weights to
`/kaggle/working/best_<backbone>_decoder.pth`.
