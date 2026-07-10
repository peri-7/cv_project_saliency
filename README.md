# cv_project_saliency

A comparison study of frozen pretrained backbones for visual saliency prediction
on the [SALICON](http://salicon.net/) dataset. Every backbone is frozen and feeds
a single shared, trainable decoder; only the decoder trains, so any difference in
score reflects the backbone's features rather than the head. The question we set
out to answer is which pretraining objective yields features most useful for
saliency.

## Backbones

We compare six ViT-B-scale backbones chosen to have roughly the same parameter
count (~86-90M) so the pretraining objective, not model capacity, is the variable.
ResNet-50 is kept as the original baseline for reference; it is not ViT-B, so it
sits outside the param-matched family.

| Backbone      | Pretraining objective         | Class in `src/models.py` |
|---------------|-------------------------------|--------------------------|
| ResNet-50     | ImageNet supervised           | `ResNet` (baseline)      |
| ViT-B         | ImageNet supervised           | `ViT` (baseline)         |
| SAM ViT-B     | Segment Anything (SA-1B)      | `SamViT`                 |
| DINOv2 ViT-B  | self-distillation             | `DinoV2ViT`              |
| DINOv3 ViT-B  | self-supervised (Gram-anchor) | `DinoV3ViT`              |
| MAE ViT-B     | masked autoencoding           | `MaeViT`                 |
| CLIP ViT-B    | image-text contrastive        | `ClipViT`                |

Every backbone follows one contract so the shared decoder consumes it unchanged:
freeze all params, lock `eval()`, return a dict of multi-scale feature tensors, and
expose an `out_channels` list. ResNet draws its four scales from four resolutions
(conv stages /4 /8 /16 /32). The plain-ViT family has only one resolution, so the
comparable multi-scale signal comes from tapping four evenly-spaced transformer
blocks (blocks 2/5/8/11 of ViT-B's 12), i.e. variety in abstraction depth rather
than spatial resolution. This four-block tap is the shared contract across the ViT
family, so fairness is structural rather than a per-model tuning choice.

A few backbones need extra plumbing. SAM's timm encoder uses windowed attention and
keeps features as spatial `[B, H, W, C]` with no CLS token, so `SamViT` only permutes
to channels-first. The rest are token-sequence models (`[B, 1+N, C]`) that need a
token-to-grid reshape and a prefix-token drop. DINOv2 is patch-14, so 480x640 is not
divisible by 14 and the input is cropped to 476x630 (a 34x45 grid) before patch
embedding. DINOv3 is patch-16 (clean 30x40 grid) but inserts register tokens between
CLS and the patch tokens, so instead of dropping one leading token it keeps the last
`H_p*W_p` tokens. The supervised ViT, MAE, and CLIP are plain patch-16 with a single
CLS token.

## Approach

Each model is a frozen backbone plus a small trainable decoder:

- The backbone is frozen and run in `eval()` mode. It emits multi-scale feature maps
  from several depths (low-level contrast up to high-level semantics).
- The decoder fuses those features and predicts a single-channel saliency map. It is
  the only part that trains, so any score difference between runs reflects the
  backbone, not the head.

For the comparison to be valid the backbone must be the only thing that changes, so
everything else is held fixed across runs: decoder `hidden_dim`, optimizer and
learning rate (`Adam`, `lr=1e-4`), epoch count / LR schedule, batch size, input
resolution (480x640), the ImageNet input normalization, and the train/val split.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/models.py` | Frozen backbones and their multi-scale feature extractors |
| `src/decoder.py` | Shared trainable decoder (`Decoder`) plus the upgraded `ConvUpDecoder` used by the LoRA experiment |
| `src/losses.py` | `Composite_Loss` = `10·KLD − CC − SIM` (KLD/SIM on softmax probs, CC on logits) |
| `src/metrics.py` | Discrete test-time metrics: NSS, AUC-Judd, Information Gain |
| `src/dataset.py` | Datasets for raw images, cached features, and end-to-end training |
| `src/training_online.py` | End-to-end train / eval / test loop (the primary route) |
| `src/training.py` | Offline two-phase loop (cache features to disk, then train the decoder) |
| `src/lora.py` | LoRA-adapted DINOv3 backbone + LoRA training loop (the ceiling experiment) |
| `local_tests/` | Local smoke tests against `mini_data/` |
| `kaggle_tests/` | Kaggle smoke tests + the decoder-width sweep used to lock `hidden_dim`=256 |
| `notebooks/` | Kaggle training scripts, one per backbone, plus real-run notebooks |
| `lora/` | LoRA + upgraded-decoder experiment: training, eval, visualization, checkpoints |
| `weight_analysis/` | Decoder interpretability (which tap depth the decoder relies on); see `WEIGHTS.md` |
| `phase1_analysis/` | Metric-metric agreement analysis (Spearman correlation between metrics) |
| `results/` | Benchmark plots and captured run logs |
| `viz_out/` | Qualitative prediction grids for the report |
| `saved_models/` | Trained decoder checkpoints |
| `scripts/mat_to_png.py` | Convert a SALICON `.mat` fixation file to a binary PNG fixation map |
| `scripts/compute_baseline.py` | Build the empirical center-bias baseline from the training fixations |
| `scripts/evaluate_ig.py` | Re-score Information Gain for every backbone against that baseline |

The final benchmark reports the full metric set: KLD, CC, SIM (continuous, from the
loss) and NSS, AUC-Judd, Information Gain (discrete, from `metrics.py`).

## Beyond the fairness study

Two follow-ups build on the roster and are documented separately:

- **LoRA + upgraded decoder** (`lora/`, `lora/LORA.md`): a separate pathway that
  unfreezes DINOv3 through LoRA adapters and swaps in a progressive-upsampling
  decoder to chase the best possible SALICON score. It is deliberately outside the
  frozen-backbone fairness study; the two upgrades are independently switchable so
  each can be attributed in an ablation.
- **Decoder interpretability** (`weight_analysis/`, `weight_analysis/WEIGHTS.md`):
  a per-tap importance analysis and a leave-one-tap-out ablation, asking which
  abstraction depth each decoder actually reads from and whether that differs across
  pretraining objectives.

## Data

Expects SALICON-style data, standardized to 480x640:

```
data/
  images/{train,val,test}/      # RGB stimuli (.jpg/.png)
  maps/{train,val,test}/        # continuous saliency maps (.png)  -> used by the loss
  fixations/{train,val,test}/   # discrete fixations (.mat)        -> used by the metrics
```

The `test` split has no maps, so final metrics are reported on `val` as a proxy.
SALICON `.mat` files store gaze as a `(num_subjects, 1)` struct; the loaders union the
fixations of all subjects to build the ground-truth fixation map.

## Setup and usage

Run from the repo root so `src/` is importable. The ViT backbones load through
[`timm`](https://github.com/huggingface/pytorch-image-models) (`pip install timm`;
preinstalled on Kaggle); ResNet only needs `torchvision`. DINOv3 pulls license-gated
weights from Hugging Face, so it needs `huggingface_hub.login(token=...)` before the
model is built.

```bash
# Online end-to-end smoke tests (backbone runs live, decoder trains).
# Run as modules from the repo root so `src/` is importable.
python -m local_tests.testing2     # ResNet baseline
python -m local_tests.testing_sam  # SAM ViT-B
python -m local_tests.testing_mae  # MAE ViT-B
python -m local_tests.testing_vit  # supervised ViT-B

# Offline two-phase smoke test: extract features to disk, then train the decoder.
python -m local_tests.testing

# Utility scripts (run as modules from the repo root).
python -m scripts.mat_to_png <input.mat> [output.png] [--radius N]  # .mat -> PNG fixation map
python -m scripts.compute_baseline                                  # build the center-bias baseline
python -m scripts.evaluate_ig                                       # re-score Information Gain
```

Real training runs use the Kaggle scripts in `notebooks/`, which point at a Kaggle
SALICON dataset path and save the best decoder weights to
`/kaggle/working/best_<backbone>_decoder.pth`.
