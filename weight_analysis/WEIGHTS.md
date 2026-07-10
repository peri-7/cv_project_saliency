# Decoder interpretability

## Goal

Interpret the trained decoder for each backbone to answer one question:

> Which abstraction depth (tap) does the decoder actually rely on, and does that
> differ across pretraining objectives?

This is a cross-backbone story. The point of the parent study is that the backbone
is the only variable; here we ask whether the kind of pretraining changes where in
the backbone the shared decoder reads its signal from. One hypothesis worth checking:
a segmentation-pretrained backbone (SAM) may lean on shallow/early taps (clean
spatial boundaries), while semantic-pretrained ones (DINOv2, CLIP) may lean on
deep/late taps.

## Vocabulary

- **tap** = a layer we extract features from. For the ViT-B family the four taps are
  transformer blocks 2, 5, 8, 11. For ResNet they are the four conv stages.
- **shallow / early / low-level** = block 2 (or ResNet stage 1): edges, textures,
  local structure.
- **deep / late / high-level** = block 11 (or ResNet stage 4): semantic, object-level.

## The decoder we are inspecting

See `src/decoder.py`. The relevant layer is the first one, `channel_compression`: a
1x1 conv, i.e. a per-pixel matrix multiply. Ignoring the trivial 1x1 spatial dims, its
weight is a matrix `W` of shape `[hidden_dim, total_in]`:

- ViT backbones: `total_in = 4 taps × 768 = 3072`, `hidden_dim = 256` → W is [256, 3072]
- ResNet: taps have unequal channels (256/512/1024/2048), `total_in = 3840`,
  `hidden_dim = 128` (we use the 128-width ResNet decoder — it beat the 256 one).

`W[i, c]` is how much output neuron `i` reads from input channel `c`. Every input
channel belongs to exactly one tap (the taps are concatenated along the channel dim in
`Decoder.forward`):

```
ViT layout:   ch    0- 767 -> tap1 (block 2, shallow)
              ch  768-1535 -> tap2 (block 5)
              ch 1536-2303 -> tap3 (block 8)
              ch 2304-3071 -> tap4 (block 11, deep)
```

(ResNet: 0-255 stage1, 256-767 stage2, 768-1791 stage3, 1792-3839 stage4.)

## Checkpoints (in `saved_models/`)

| Checkpoint                   | Backbone        | hidden_dim |
|------------------------------|-----------------|------------|
| `best_resnet_decoder128.pth` | ResNet-50       | 128        |
| `best_sam_decoder.pth`       | SAM ViT-B       | 256        |
| `best_mae_decoder.pth`       | MAE ViT-B       | 256        |
| `best_dinov2_decoder.pth`    | DINOv2 ViT-B/14 | 256        |
| `best_dinov3_decoder.pth`    | DINOv3 ViT-B/16 | 256        |

Also present but not used here: `best_resnet_decoder.pth` (256),
`best_sam_decoder128.pth`. CLIP and supervised ViT are analyzed with their own
checkpoints when available. The Kaggle checkpoint path is an editable variable at the
top of each per-model script.

## Analysis A — per-tap importance heatmap (correlational)

Build a matrix `M` of shape `[hidden_dim, 4]`:

```
M[i, k] = sum over channels c in tap k of  |W[i, c]| · std(x_c)
```

`std(x_c)` is the standard deviation of input channel `c`, measured over a forward
pass of the frozen backbone on the val set. This scale correction is essential: raw
`|W|` is misleading because a weight's real effect depends on the magnitude of the
feature it multiplies (a tap with tiny features needs big weights just to matter).
`|W[i,c]| · std(x_c)` is the typical size of the contribution that channel makes.
`channel_compression` has `bias=False` and is followed by GroupNorm, so absolute scale
is partly absorbed downstream — only relative (per-tap) comparison is meaningful, which
is exactly what we want.

Why `|W|·std` and not raw `|W|`: a neuron's output is `Σ_c W[i,c]·x_c`, so a channel's
contribution depends on weight *and* feature size. Channels live at very different
magnitudes (early vs late blocks, ResNet stage1 vs stage4), and training learns
`|W| ∝ 1/magnitude`: small-but-useful channels get large weights, large ones get small
weights. So raw `|W|` mostly measures how small a channel's features were, not how much
the decoder relies on it — and these runs use plain Adam (no weight decay), so even
dead weights are not pruned to zero. Multiplying by `std(x_c)` cancels the scale and
leaves the typical contribution size, which is what we mean by reliance.

Why `std` and not `mean`/`max`: what shapes the saliency map is how much a channel
varies across positions/images, not its baseline level. The mean is a constant offset
that shifts the whole map uniformly and is re-centered away by the downstream GroupNorm,
so it does not change the pattern. `max` keys off a single outlier pixel and is noisy;
we want the typical swing.

Plot: heatmap, rows = output neurons (256, or 128 for ResNet), columns = the 4 taps
(shallow → deep). Two readings: column sums show which tap the decoder relies on
overall (the headline); reading across each row (sorted by dominant tap) shows whether
neurons specialize by depth (banding) or blend everything. We collapse the full
`[hidden_dim, 3072]` matrix to 4 tap-columns because individual channel indices are not
comparable across backbones, but the tap is.

## Analysis B — leave-one-tap-out ablation (causal)

Confirms that A's bright columns actually matter. Compute a baseline pass over val with
nothing zeroed (the full 7-tuple `test_model_online` returns), then 4 passes each
zeroing one tap's feature tensor before the decoder, and recompute the 7-tuple. Report
all metrics as deltas vs baseline per zeroed tap. The tap whose removal hurts most is
load-bearing, and it should agree with A's dominant column.

We skip gradient×activation attribution — A and B tell the same story for a near-linear
fusion head, with less machinery.

## Code structure

```
weight_analysis/
    WEIGHTS.md         # this file
    utils.py           # shared: tap_importance() + ablation_study() via run_analysis()
    resnet_anal.py     # instantiate backbone + decoder, load ckpt, call utils
    sam_anal.py
    mae_tap_importance.csv, ...  # per-backbone outputs
    plot_cross_backbone.py       # cross-backbone depth figures from the CSVs
```

Per-model scripts are thin (~20 lines): build the right backbone, build `Decoder` with
the right `hidden_dim` and `extractor.out_channels`, `load_state_dict`, then run A and
B. All A+B logic lives in `utils.py`. Run from the repo root as modules
(`python -m weight_analysis.resnet_anal`). The scripts target Kaggle for the real run
(GPU + full val set), mirror the data setup from `notebooks/dinov2.py`, and hold the
transforms fixed for every backbone (fairness rule). Tap-channel boundaries come from
`extractor.out_channels`, so ResNet's unequal stages and DINOv2's grid size are handled
by the same code.

Per-backbone gotchas: ResNet has `hidden_dim=128` and unequal tap channels at different
resolutions (the std-weighting handles the scale). DINOv2 is patch-14, so its grid is
34x45 not 30x40 — transparent to the decoder. DINOv3 keeps the last `H_p*W_p` tokens
internally, so its emitted feature dict is a normal `[B,768,30,40]`. All backbones are
frozen and run under `torch.no_grad()` (both analyses are inference-only).

## Results

### ResNet baseline (`best_resnet_decoder128.pth`, hidden_dim=128, full val)

Analysis A — tap importance. Raw column sums grow monotonically with depth, but
ResNet's taps have unequal channel counts, so the column sum is partly just "more
channels":

| tap          | chans | column sum | share | per-channel (sum/chans) |
|--------------|-------|------------|-------|-------------------------|
| stage1 (/4)  | 256   | 275.1      | 11.1% | 1.075                   |
| stage2 (/8)  | 512   | 414.2      | 16.7% | 0.809                   |
| stage3 (/16) | 1024  | 766.5      | 31.0% | 0.749                   |
| stage4 (/32) | 2048  | 1019.6     | 41.2% | 0.498                   |

Per channel the order reverses (shallow highest), and stage3 > stage4.

Analysis B — ablation (Δ vs baseline; baseline loss 1.571). Every metric agrees on one
ranking: stage3 ≫ stage4 > stage2 > stage1. Zeroing stage3 nearly doubles the loss
(→3.19), ~2.3x the damage of zeroing stage4.

| zeroed tap       | Δloss | Δcc    | Δnss   | Δig    |
|------------------|-------|--------|--------|--------|
| stage1 (/4)      | +0.29 | −0.008 | −0.017 | −0.060 |
| stage2 (/8)      | +0.53 | −0.019 | −0.034 | −0.090 |
| **stage3 (/16)** | +1.62 | −0.060 | −0.102 | −0.235 |
| stage4 (/32)     | +0.70 | −0.037 | −0.075 | −0.101 |

Reading: A's raw column-sum headline ("stage4 dominates") is a channel-count artifact,
not real reliance. Correct A for channel count and it flips to agree with the causal
ablation: stage3 (/16, mid/high level) is the load-bearing tap, not the deepest stage.
The two shallow taps matter least in both analyses. Per-neuron rows show no banding —
nearly every neuron has the same monotone profile (stage4 ≥ stage3 > stage2 > stage1),
so the decoder blends taps homogeneously rather than specializing neurons by depth.

Caveat for the writeup: lead with the ablation (or per-channel-normalized A) for
ResNet. The confound is ResNet-specific — the ViT family has 768 channels per tap
(equal), so their column sums are directly comparable across taps and this does not
bite.

### SAM ViT-B (`best_sam_decoder.pth`, hidden_dim=256, full val)

The clean case: all four taps share 768 channels and one /16 grid, so there is no
channel-count confound and no resolution difference between taps. A and B agree cleanly.

Analysis A — tap importance (column sums directly comparable; per-channel is just ÷768):

| tap                | column sum | share |
|--------------------|------------|-------|
| block 2 (/16)      | 1705.6     | 15.4% |
| block 5 (/16)      | 2163.9     | 19.5% |
| block 8 (/16)      | 3001.4     | 27.1% |
| **block 11 (/16)** | 4216.3     | 38.0% |

Analysis B — ablation (Δ vs baseline; baseline loss 0.769). Ranking
block 11 ≫ block 8 > block 2 ≈ block 5:

| zeroed tap         | Δloss  | Δcc     | Δnss    | Δig    |
|--------------------|--------|---------|---------|--------|
| block 2 (/16)      | +0.350 | +0.0001 | +0.011  | −0.037 |
| block 5 (/16)      | +0.306 | −0.004  | +0.0001 | −0.030 |
| block 8 (/16)      | +0.607 | −0.036  | −0.074  | −0.094 |
| **block 11 (/16)** | +1.044 | −0.077  | −0.168  | −0.198 |

Reading: importance rises monotonically with depth, block 11 dominant, and A and B
agree with no correction needed (the clean validation that the two methods converge
absent a confound). The two shallow taps are nearly free to remove: zeroing block 2
leaves cc unchanged (+0.0001) and even raises NSS (+0.011). The decoder reads almost
entirely from the deep half (blocks 8 + 11). Since all taps are /16, this is pure
abstraction-depth preference, not a resolution effect. No banding (every neuron
monotone block11 > … > block2).

Headline — this contradicts the hypothesis. We guessed SAM (segmentation-pretrained)
would lean shallow (clean boundaries). It does the opposite: it leans hardest on its
deepest block and its shallow blocks are near dead weight. More interesting than
confirming the guess.

ResNet vs SAM contrast (handle with care): ResNet's load-bearer is mid-level (stage3),
SAM's is deepest (block 11). But ResNet's stages differ in resolution, so "mid wins"
could partly be the deepest stage being penalized for /32 coarseness, not for
abstraction. SAM's blocks are all /16, so depth is isolated cleanly. The fair
cross-model comparison is therefore SAM vs the other all-/16 ViTs (MAE/DINOv2/DINOv3);
ResNet stays the resolution-confounded baseline.

## Future work

Add a per-channel-normalized view to Analysis A so the channel-count confound is
visible automatically (and harmless for the equal-width ViTs): alongside each tap's
column sum, also report `column_sum / out_channels[k]` (mean contribution per channel).
For ResNet the two columns tell different stories; for the ViTs they are proportional.
A later spatial extension could ablate tap `k` and visualize how the output saliency
map changes, showing where each depth contributes.

## Notes

- Use `best_resnet_decoder128.pth` (128-width) as the ResNet baseline; its heatmap has
  fewer rows (128), which is fine for a reference.
- Report all 7 metrics in the ablation.
