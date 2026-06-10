# Decoder interpretability — design doc

This doc captures everything decided in the planning conversation so work can resume
in a fresh session. Read this + `CLAUDE.md` and you have full context.

## Goal

Interpret the trained **decoder** for each backbone to answer one question:

> **Which abstraction depth (tap) does the decoder actually rely on, and does that
> differ across pretraining objectives?**

This is a cross-backbone story. The point of the parent study is that the backbone is
the only variable; here we ask whether the *kind* of pretraining changes *where in the
backbone* the shared decoder reads its signal from. Hypothesis worth checking: a
segmentation-pretrained backbone (SAM) may lean on shallow/early taps (clean spatial
boundaries), while semantic-pretrained ones (DINOv2, CLIP) may lean on deep/late taps.

## Vocabulary (fixed)

- **tap** = a layer we extract features from. For the ViT-B family the four taps are
  transformer **blocks 2, 5, 8, 11**. For ResNet they are the **four conv stages**.
- **shallow / early / low-level** = block 2 (or ResNet stage 1): edges, textures, local
  structure.
- **deep / late / high-level** = block 11 (or ResNet stage 4): semantic, object-level.

## The decoder we're inspecting

See `src/decoder.py`. The relevant layer is the FIRST one, `channel_compression`:
a **1×1 conv**, i.e. a per-pixel matrix multiply. Ignoring the trivial 1×1 spatial
dims, its weight is a matrix:

```
W  with shape  [hidden_dim, total_in]
```

- ViT backbones: `total_in = 4 taps × 768 = 3072`, `hidden_dim = 256`  → W is [256, 3072]
- ResNet: taps have unequal channels (256/512/1024/2048), `total_in = 3840`,
  `hidden_dim = 128` (we use the 128-width ResNet decoder — it beat the 256 one).

`W[i, c]` = how much output neuron `i` reads from input channel `c`. Every input channel
belongs to exactly one tap (the taps are simply concatenated along the channel dim in
`Decoder.forward`):

```
ViT layout:   ch    0– 767 → tap1 (block 2, shallow)
              ch  768–1535 → tap2 (block 5)
              ch 1536–2303 → tap3 (block 8)
              ch 2304–3071 → tap4 (block 11, deep)
```

(ResNet: 0–255 stage1, 256–767 stage2, 768–1791 stage3, 1792–3839 stage4.)

## Checkpoints (in `saved_models/`)

| Checkpoint                  | Backbone        | hidden_dim |
|-----------------------------|-----------------|------------|
| `best_resnet_decoder128.pth`| ResNet-50       | **128**    |
| `best_sam_decoder.pth`      | SAM ViT-B       | 256        |
| `best_mae_decoder.pth`      | MAE ViT-B       | 256        |
| `best_dinov2_decoder.pth`   | DINOv2 ViT-B/14 | 256        |
| `best_dinov3_decoder.pth`   | DINOv3 ViT-B/16 | 256        |

(Also present but NOT used: `best_resnet_decoder.pth` (256), `best_sam_decoder128.pth`.)
CLIP and supervised ViT have **no checkpoint yet** — skip them for now.

The user handles the Kaggle checkpoint-loading path themselves — the scripts should make
the checkpoint path an obvious, easy-to-edit variable at the top, not hard-coded.

## Analysis A — per-tap importance heatmap (correlational)

Build a matrix `M` of shape `[hidden_dim, 4]`:

```
M[i, k] = sum over channels c in tap k of  |W[i, c]| · std(x_c)
```

- `std(x_c)` = standard deviation of input channel `c`, measured over a forward pass of
  the **frozen backbone on the val set**. This scale correction is essential: raw
  `|W|` is misleading because a weight's real effect depends on the magnitude of the
  feature it multiplies (a tap with tiny features needs big weights just to matter).
  `|W[i,c]| · std(x_c)` is the *typical size of the contribution* that channel makes.
- Note `channel_compression` has `bias=False` and is followed by `GroupNorm`, so absolute
  scale is partly absorbed downstream — **only relative (per-tap) comparison is
  meaningful**, which is exactly what we want.

**Plot:** heatmap, rows = output neurons (256, or 128 for ResNet — fewer rows is fine,
ResNet is just the baseline), columns = the 4 taps (shallow→deep). Two readings:

- **Column sums** → which tap the decoder relies on overall. **This is the headline.**
- **Across each row** (sort rows by dominant tap) → whether neurons *specialize* by depth
  (banding) vs. blending everything.

We deliberately collapse the full `[hidden_dim, 3072]` matrix to 4 tap-columns: individual
channel indices aren't comparable across backbones, the tap is. Per-channel "zoom in" on a
single interesting tap (768 cols) is a possible follow-up, not first pass.

## Analysis B — leave-one-tap-out ablation (causal)

Confirms that A's bright columns actually matter.

- Compute a **baseline** pass over val with nothing zeroed → full 7-tuple
  `(loss, kld, cc, sim, nss, auc, ig)` (same tuple `test_model_online` returns).
- Then **4 passes**, each zeroing one tap's feature tensor before the decoder, and
  recompute the 7-tuple.
- **Report all metrics** (user wants the full set, not one headline). Output: a table of
  deltas vs baseline per zeroed tap. The tap whose removal hurts most is load-bearing;
  it should agree with A's dominant column.

Optional later extension: spatial ablation-difference maps (ablate tap k, visualize how
the output saliency map changes) — shows *where* each depth contributes. Not first pass.

We are **skipping** gradient×activation attribution — A+B tell the same story for a
near-linear fusion head, with less machinery.

## Code structure

```
weight_analysis/
    PLAN.md            # this file
    utils.py           # shared: tap_importance_heatmap() + ablation_study()
    resnet.py          # instantiate backbone + decoder, load ckpt, call utils
    sam.py
    mae.py
    dinov2.py
    dinov3.py
```

- Per-model scripts are thin (~20 lines): build the right backbone, build `Decoder` with
  the right `hidden_dim` and `extractor.out_channels`, `load_state_dict`, run A then B.
- All A+B logic lives in `utils.py` so it's not copy-pasted five times.
- Run from repo root as modules, same pattern as `local_tests/` and the import contract
  in `CLAUDE.md`: `python -m weight_analysis.resnet`.
- The scripts target **Kaggle** for the real run (GPU + full val set). Mirror the data
  setup from `notebooks/dinov2.py`:
  - transforms: `Resize((480,640))` + `ToTensor` + ImageNet `Normalize` (held fixed for
    every backbone, including the analysis — fairness rule).
  - `LoraDataset` for the **val** split; data root `/kaggle/input/datasets/roshan401/salicon`.
  - figures/tables saved to `/kaggle/working/`.
- Tap-channel boundaries come from `extractor.out_channels` (don't hard-code 768/3072) so
  ResNet's unequal stages and DINOv2's grid size are handled by the same code.

## Per-backbone gotchas (from CLAUDE.md / src/models.py)

- **ResNet**: `hidden_dim=128`; unequal tap channels (256/512/1024/2048) at different
  spatial resolutions. A's per-tap sum is over different channel counts — that's fine, the
  std-weighting handles scale; just don't assume 768 per tap.
- **DINOv2**: patch-14, input cropped to 476×630 → **34×45** feature grid (others are
  30×40). Transparent to the decoder; only matters if any code assumes 30×40.
- **DINOv3**: patch-16, clean 30×40, but token sequence is `[CLS, reg…, patch…]` — the
  model already keeps the last `H_p*W_p` patch tokens internally, so the emitted feature
  dict is normal `[B,768,30,40]`. No special handling needed here.
- All backbones are frozen + locked to `eval()`; run the backbone in `torch.no_grad()`
  (the std-collection pass and both analyses are inference-only — nothing trains).

## Decisions locked

- Use `best_resnet_decoder128.pth` (128-width) as the ResNet baseline.
- ResNet heatmap has fewer rows (128) — acceptable, it's just the baseline reference.
- Kaggle checkpoint path: user handles it; keep it an editable top-of-file variable.
- Report all 7 metrics in B.
- CLIP + supervised ViT: no checkpoints → not analyzed for now.
```
