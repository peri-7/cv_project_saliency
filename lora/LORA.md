# LoRA + upgraded decoder experiment

This experiment uses DINOv3 ViT-B with LoRA adapters on the backbone plus an
upgraded convolutional decoder, aiming for the best possible SALICON saliency
score. It is deliberately outside the frozen-backbone fairness study: the roster
keeps its frozen backbones and minimal decoder untouched, and this run lives in its
own files.

The two upgrades (LoRA, new decoder) are independently switchable, so we can run
them together for the ceiling result and then flip one flag at a time for
attribution.

## Design decisions

1. **DINOv3 backbone.** `DinoV3ViT` already exists in `src/models.py`
   (`vit_base_patch16_dinov3`, patch-16, register tokens, keeps the last
   `H_p*W_p` tokens).

2. **LoRA rather than unfreezing the last N blocks.** LoRA is the cheaper option,
   not just the fancier one. The dominant memory cost of any backbone tuning is the
   activation memory for backprop through the trunk, which is identical either way;
   LoRA just adds far fewer trainable/optimizer params on top. The lever for fitting
   a 16 GB Kaggle T4/P100 is gradient checkpointing on the backbone plus a smaller
   batch (16 → 4-8), not avoiding LoRA.

3. **Manual LoRA, not `peft`.** A ~40-line `LoRALinear` keeps Kaggle install-free
   and matches the repo's plain-scripts style. The math is the same either way
   (frozen base + trainable low-rank `A·B` adapter). Init is kaiming on `lora_A` and
   zeros on `lora_B` so the adapter starts as a no-op. Swapping to `peft` later would
   not change the `LoraDinoV3ViT` interface.

4. **AdamW, not plain Adam.** There is no reason to prefer plain Adam over AdamW's
   decoupled weight decay, so both the decoder-only and LoRA+decoder runs use AdamW.
   The base `notebooks/dinov3.py` was switched from Adam to AdamW too, to keep runs
   comparable.

5. **Upgraded convolutional decoder, not a transformer decoder.** The core upgrade
   is distributing the upsampling across multiple learned stages instead of one big
   bilinear stretch at the end:

   - The old `Decoder` fuses the four /16 taps, runs two conv layers, outputs a logit
     at /16 (30x40), and then the training loop does one bilinear jump all the way to
     480x640 — a 16x magnification with no learned refinement.
   - The new `ConvUpDecoder` fuses the four /16 taps, refines at /16, then
     progressively upsamples /16 → /8 → /4 with a learned conv block at each stage
     that can sharpen detail before the next stretch. The final bilinear jump is then
     only 4x, so there is much less blur to fix and the result is sharper.

   This is a progressive-upsampling (SETR-PUP-style) decoder, not FPN. FPN needs a
   resolution pyramid (/4, /8, /16, /32) so it can fuse across scales with lateral
   connections; all four DINOv3 taps are at the same /16 grid (they differ in
   abstraction depth, not spatial resolution), so there is no pyramid to ladder over.
   A transformer decoder was rejected because the trunk already did the global
   reasoning, attention at /8 or /4 is memory-prohibitive on a T4 also running LoRA
   backprop, and a single smooth saliency heatmap has no object-query structure to
   exploit. Relevant work: DPT (Ranftl et al., ICCV 2021), SETR-PUP/MLA (Zheng et
   al., CVPR 2021), ViT-Adapter (Chen et al., ICLR 2022), DINOv2 dense heads (Oquab
   et al., 2023); saliency-specific: TranSalNet (Lou et al., Neurocomputing 2022) and
   DeepGaze IIE (Linardos et al., ICCV 2021).

6. **Everything else stays fixed** per project convention: `hidden_dim=256`, 480x640
   input, ImageNet normalization, the `Composite_Loss` (10·KLD − CC − SIM), the
   7-metric contract, and the val-as-test proxy.

## What lives where

### `src/decoder.py` — `ConvUpDecoder`

Drop-in compatible with `Decoder`: same constructor signature
(`in_channels_list`, `hidden_dim=256`), same dict-or-list forward input, returns raw
logits `[B, 1, H/4, W/4]` (the train/eval loops already interpolate any decoder output
up to GT size, so a /4 output is fine). Architecture: fuse the four /16 taps
(concat → 1x1 conv → `hidden_dim`, GroupNorm, ReLU); refine at /16 with a 3x3 conv
block; progressively upsample with resize-then-conv (bilinear + 3x3, which avoids the
transposed-conv checkerboard) tapering channels /16 → /8 → /4; then a 1x1 logit head
at /4. GroupNorm throughout (small batches), reusing the existing "largest #groups ≤ 32
that divides channels" divisor logic.

### `src/lora.py` — LoRA layer + LoRA-enabled DINOv3 + LoRA train loop

- **`LoRALinear(base_linear, r=8, alpha=16, dropout=0.0)`** wraps an existing
  `nn.Linear`, freezes its weight/bias, and adds `lora_A` (in→r, kaiming) and
  `lora_B` (r→out, zeros) so the adapter starts as a no-op. `scaling = alpha / r`;
  `forward(x) = base(x) + scaling * lora_B(lora_A(dropout(x)))`.

- **`LoraDinoV3ViT`** loads DINOv3 the same way `DinoV3ViT` does (four taps at blocks
  2/5/8/11, keep the last `H_p*W_p` tokens, reshape to `[B, 768, 30, 40]`), but:
  freezes all base params and injects `LoRALinear` into each block's attention
  (`qkv`, optionally `proj`); runs the trunk *with* grad (no `torch.no_grad()` around
  the forward, which is what would otherwise kill LoRA gradients); does not force
  eval mode in `train()` (ViT has only LayerNorm, no BN stats to corrupt, and we want
  dropout active); and exposes gradient checkpointing via
  `set_grad_checkpointing(True)` to fit batch ≈ 8 on 16 GB. It is parametric over
  `lora_blocks` (which blocks get adapters; default all 12), `tap_blocks` (which
  blocks features are read from; default 2/5/8/11), `lora_targets`, `r`, `alpha`, and
  `lora_dropout`.

  It reads features with timm's `forward_intermediates()` rather than forward hooks.
  Under gradient checkpointing a block's forward runs in no-grad mode (recomputed
  during backward), so a hook would capture a tensor detached from the graph and LoRA
  would silently train at zero gradient. `forward_intermediates()` returns the
  checkpoint outputs, which keep their `grad_fn`. There is also a
  `requires_grad_(True)` guard on the patch-embed output and a first-batch assertion
  that LoRA gradients are non-zero, which fails loudly if features ever come back
  detached.

- **`train_one_epoch_lora`** mirrors `train_one_epoch_online` but runs the backbone
  forward with grad and optimizes decoder + LoRA params together, returning the same
  `(avg_loss, avg_kld, avg_cc)` tuple. Validation and final benchmarking reuse
  `evaluate_model_online` / `test_model_online` unchanged (they run under
  `torch.no_grad()`, which is correct at inference).

- **`test_model_online_lora`** returns the same 7-tuple as `test_model_online` but
  computes Information Gain against a Gaussian center-bias baseline (`sigma = 0.25` of
  each image dimension) rather than the uniform map. Its IG is therefore not directly
  comparable to the frozen-roster IG (uniform baseline); the other six metrics are.

### Training scripts (`lora/`)

`dino3_lora.py` is the main training script with two flags at the top:

```python
USE_LORA = True          # True -> LoraDinoV3ViT + train_one_epoch_lora; False -> frozen DinoV3ViT
DECODER  = "upgraded"    # "upgraded" -> ConvUpDecoder; "baseline" -> Decoder
```

`dino3_lora_2tap.py` is the same script with `tap_blocks=[10, 11]` (see the
weight-analysis result that blocks 7 and 8 are near-redundant). The `*_eval.py`
scripts re-run only the final evaluation on a saved checkpoint, for when a Kaggle
session times out before the evaluation block runs. `visualize_predictions.py` and
`visualize_comparison.py` produce the qualitative figures for the report.

## Ablation plan

| Run      | USE_LORA | DECODER  | Isolates                            |
|----------|----------|----------|-------------------------------------|
| Ceiling  | True     | upgraded | both upgrades together              |
| −LoRA    | False    | upgraded | decoder-only gain vs frozen DINOv3  |
| −decoder | True     | baseline | LoRA-only gain vs frozen DINOv3     |
| baseline | False    | baseline | matches `notebooks/dinov3.py`       |
