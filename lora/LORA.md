# LoRA + Upgraded Decoder experiment — design & decisions

This document captures the plan agreed in a planning conversation, so a fresh
conversation can implement it without re-deriving anything. Read this together
with the repo's top-level `CLAUDE.md`.

## Goal

Build a training pathway that uses **DINOv3 ViT-B + LoRA** on the backbone, plus
an **upgraded convolutional decoder**, to push for the **best possible SALICON
saliency score** — explicitly NOT part of the frozen-backbone *fairness study*.
This is a separate experiment (hence its own files); the fairness roster keeps
its frozen backbones and minimal decoder untouched.

The two upgrades (LoRA, new decoder) must be **independently switchable** so we
can run them together for the ceiling result, then flip one flag at a time for
attribution/ablation. Switchable is a strict superset: with both flags on we get
the combined run; the ablations are the same script with one flag changed.

## Decisions made

1. **Use DINOv3** (`DinoV3ViT` already exists in `src/models.py`,
   `vit_base_patch16_dinov3`, patch-16, register tokens → keep the last
   `H_p*W_p` tokens).

2. **LoRA over unfreezing last N blocks.** LoRA is the *cheaper* option, not just
   the fancier one: the dominant memory cost of any backbone tuning is the
   activation memory for backprop through the trunk, which is identical either
   way; LoRA just adds far fewer trainable/optimizer params on top. The lever for
   fitting a Kaggle T4/P100 (16 GB) is **gradient checkpointing on the backbone +
   a smaller batch** (drop 16 → 4–8), NOT avoiding LoRA.

3. **Manual LoRA implementation** (a ~40-line `LoRALinear`), NOT the `peft`
   library — keeps Kaggle install-free and matches the repo's plain-scripts
   ethos. Same math either way (frozen base + trainable low-rank `A·B` adapter);
   `peft` would auto-inject adapters via `LoraConfig(target_modules=[...])` and
   add save/merge/print helpers, at the cost of an extra dependency we don't need
   for adapting just `qkv`/`proj` on 12 blocks. Easy to swap to `peft` later (the
   `LoraDinoV3ViT` interface wouldn't change). Manual init must be kaiming on
   `lora_A` and **zeros on `lora_B`** so the adapter starts as a no-op.

6. **Optimizer: `AdamW`** (not `Adam`) — there's no reason to prefer plain `Adam`
   over `AdamW`'s decoupled weight decay. Use `AdamW` for both the decoder-only
   and the LoRA+decoder runs. (Also fix the base `notebooks/dinov3.py`, which
   currently uses plain `Adam`, to keep runs comparable.)

4. **Upgraded convolutional decoder**, NOT a transformer decoder. The core upgrade
   is **distributing the upsampling across multiple learned stages** instead of one
   big dumb bilinear stretch at the end:

   - **Old `Decoder`:** fuses the four /16 taps, runs two conv layers, outputs a
     logit at /16 (30×40), then the training loop does one bilinear jump all the
     way to 480×640 — a 16× magnification with zero learned refinement.
   - **New `ConvUpDecoder`:** fuses the four /16 taps, refines at /16, then
     progressively upsamples /16→/8→/4 with a learned conv block at each stage
     that can sharpen detail before the next stretch, output at /4 (120×160),
     then the training loop does only a 4× final jump.

   Each intermediate conv block can fix blurring artifacts introduced by the
   bilinear upsample before they propagate to the next stage. By the time the
   final bilinear happens it's 4× not 16×, so there's much less to fix and the
   result is sharper.

   Note on topology: this is a **progressive upsampling / SETR-PUP-style** decoder,
   NOT FPN. Classic FPN requires a *resolution pyramid* (features at /4, /8, /16,
   /32) so it can fuse across scales with lateral connections. All four DINOv3 taps
   are at the *same* /16 grid (they differ in abstraction depth, not spatial
   resolution), so there is no pyramid to pyramid over — FPN's lateral-merge
   mechanism would be degenerate here.

   Transformer decoders were also rejected: the trunk already did the global
   reasoning; attention at /8 (4,800 tokens) or /4 (19,200 tokens) is
   memory-prohibitive on a T4 also running LoRA backprop; and saliency (one smooth
   heatmap) has no object-query structure to exploit. Supporting literature:
   **DPT** (Ranftl et al., ICCV 2021), **SETR-PUP/MLA** (Zheng et al., CVPR 2021),
   **ViT-Adapter** (Chen et al., ICLR 2022), **DINOv2** dense heads (Oquab et al.,
   2023); saliency-specific: **TranSalNet** (Lou et al., Neurocomputing 2022,
   transformer encoder + CNN decoder on SALICON) and **DeepGaze IIE** (Linardos et
   al., ICCV 2021, frozen backbone + conv readout).

5. **Keep everything else fixed** per project convention where it still applies:
   `hidden_dim=256`, 480×640 input, ImageNet normalization, the
   `Composite_Loss` (10·KLD − CC − SIM), the 7-tuple metric contract, val-as-test
   proxy.

## `notebooks/dinov3.py` review (findings from this conversation)

Structurally correct and mirrors the other backbone notebooks. Flags to be aware
of (not blockers for the LoRA work, but worth fixing in the base notebook):

- **Gated weights:** `vit_base_patch16_dinov3` pulls license-gated weights from
  Hugging Face — needs `huggingface_hub.login(token=...)` before
  `timm.create_model(pretrained=True)`, else the download 401s. **We have an
  `HF_TOKEN`** (store it as a Kaggle secret and log in at the top of the
  notebook). None of the other roster backbones need this.
- **Early-stopping dead logic:** `patience > 3` with only `epochs=10` can
  essentially never fire meaningfully. Harmless.
- **Optimizer:** line 68 uses plain `Adam`. Switch it to `AdamW` (decision #6).

## File plan (to be implemented)

### 1. `src/decoder.py` — ADD `ConvUpDecoder` (keep existing `Decoder` untouched)

Drop-in compatible with `Decoder`: same constructor signature
(`in_channels_list`, `hidden_dim=256`), same dict-or-list `forward` input,
returns raw logits `[B, 1, H/4, W/4]` (the train/eval loops already
`F.interpolate` any decoder output up to GT size, so a /4 output is fine and the
GT resolution stays immutable).

Architecture:
1. Fuse the four /16 taps: concat → 1×1 conv → `hidden_dim`, GroupNorm, ReLU.
2. Refine at /16: 3×3 conv block.
3. Progressive upsample with channel taper, using **resize-then-conv** (bilinear
   `F.interpolate(scale_factor=2)` + 3×3 conv — avoids transposed-conv
   checkerboard):
   - /16 → /8: `hidden_dim` → `hidden_dim/2`
   - /8 → /4: `hidden_dim/2` → `hidden_dim/4`
4. Logit head: 1×1 conv → 1 channel at /4.

Use GroupNorm everywhere (small-batch rationale). Reuse the existing
"largest #groups ≤ 32 that divides channels" divisor logic so any width
constructs. Helper `_ConvBlock` = Conv3×3(bias=False) → GroupNorm → ReLU.

(The exact code for this was drafted in the planning conversation and is ready to
paste; re-derive from the description above if not present.)

### 2. `src/lora.py` — NEW module: LoRA layer + LoRA-enabled DINOv3 + LoRA train loop

- **`LoRALinear(base_linear, r=8, alpha=16, dropout=0.0)`**: wraps an existing
  `nn.Linear`, freezes its weight/bias, adds `lora_A` (in→r, kaiming init) and
  `lora_B` (r→out, zero init) so the adapter starts as a no-op. `scaling =
  alpha / r`. `forward(x) = base(x) + scaling * lora_B(lora_A(dropout(x)))`.
  Needs `import math`.

- **`LoraDinoV3ViT`**: same construction and token handling as `DinoV3ViT`
  (load `vit_base_patch16_dinov3`, `dynamic_img_size=True`, four taps at blocks
  2/5/8/11, keep the **last `H_p*W_p` tokens** to skip CLS + register tokens,
  reshape → permute to `[B, 768, 30, 40]`, `out_channels=[768]*4`). DIFFERENCES
  from the frozen version:
  - Freeze ALL base params, then inject `LoRALinear` into each block's attention
    (`block.attn.qkv`, optionally `block.attn.proj`). Only the LoRA params are
    trainable.
  - **Do NOT wrap the forward in `torch.no_grad()`** — that is what kills LoRA
    gradients. The frozen `DinoV3ViT.forward` wraps the trunk in `no_grad`
    (`src/models.py`); the LoRA variant must run the trunk WITH grad. Hooks
    capturing block outputs retain grad fine.
  - **Do NOT override `train()` to force eval.** ViT has only LayerNorm (no BN
    running stats to corrupt), and we want train mode so LoRA dropout works.
  - Expose **gradient checkpointing**: call timm's
    `base_model.set_grad_checkpointing(True)` behind a constructor flag (needed
    to fit batch≈8 on a 16 GB GPU).
  - Helper to collect trainable params (`p for p in model.parameters() if
    p.requires_grad`) for the optimizer.

- **`train_one_epoch_lora(extractor, decoder, loader, optimizer, criterion,
  device)`**: like `train_one_epoch_online` in `src/training_online.py` BUT the
  backbone forward runs WITH grad (no `torch.no_grad()` around
  `extractor(images)`), and the optimizer covers both decoder AND LoRA params.
  Returns the same `(avg_loss, avg_kld, avg_cc)` 3-tuple. Set `extractor.train()`
  and `decoder.train()`.
  - **Reuse `evaluate_model_online` and `test_model_online` from
    `src/training_online.py` unchanged** for val/test — they run under
    `torch.no_grad()`, which is correct at inference for LoRA too. Only the
    training step needs the grad-enabled variant.

NOTE: this is a deliberately separate pathway from `training.py` /
`training_online.py` (the CLAUDE.md "mirror into both" rule is about those two;
LoRA is a distinct mode). Document that in the file.

### 3. `notebooks/lora.py` — NEW Kaggle training mirror (mirrors `notebooks/dinov3.py`)

Two flags at the top, BOTH ON by default (the combined ceiling run):

```python
USE_LORA = True          # True -> LoraDinoV3ViT + train_one_epoch_lora; False -> frozen DinoV3ViT + train_one_epoch_online
DECODER  = "upgraded"    # "upgraded" -> ConvUpDecoder; "baseline" -> Decoder
```

Wiring:
- `if USE_LORA`: `extractor = LoraDinoV3ViT(grad_checkpointing=True)`, train with
  `train_one_epoch_lora`, **`AdamW`** over `decoder.parameters()` + trainable LoRA
  params (consider two param groups: decoder lr 1e-4, LoRA lr ~5e-5). Else: frozen
  `DinoV3ViT`, `train_one_epoch_online`, **`AdamW`** over decoder only (lr 1e-4).
- `Decoder` vs `ConvUpDecoder` from `src/decoder.py`, both `hidden_dim=256`.
- Smaller batch when `USE_LORA` (e.g. 8 instead of 16) for memory.
- Save BOTH decoder and (if LoRA) the LoRA params to the checkpoint; benchmark
  the BEST checkpoint with `test_model_online` (full 7-tuple).
- Kaggle preamble: clone repo, add to `sys.path`, then
  `from huggingface_hub import login; login(token=<HF_TOKEN secret>)` BEFORE
  building the extractor (gated DINOv3 weights — we have the token).

## Ablation plan (after the combined run)

| Run | USE_LORA | DECODER  | Isolates |
|-----|----------|----------|----------|
| Ceiling | True  | upgraded | both upgrades together |
| −LoRA   | False | upgraded | decoder-only gain vs frozen dinov3 |
| −decoder| True  | baseline | LoRA-only gain vs frozen dinov3 |
| baseline| False | baseline | == existing `notebooks/dinov3.py` |

## Status

IMPLEMENTED (2026-06-11): `ConvUpDecoder` in `src/decoder.py`, `LoRALinear` /
`LoraDinoV3ViT` / `train_one_epoch_lora` / `make_gaussian_center_bias` /
`test_model_online_lora` in `src/lora.py`, and the flagged Kaggle mirror
`notebooks/lora.py`. `notebooks/dinov3.py` was also fixed (Adam → AdamW, HF
login added). Not yet run on Kaggle.

`test_model_online_lora` returns the same 7-tuple as `test_model_online` but
computes Information Gain against a **Gaussian center-bias** baseline
(`sigma = 0.25` of each image dimension) instead of the uniform map, by
delegating to `test_model_online(baseline_prob=...)`. The lora notebook uses
it for the final benchmark — its IG is therefore NOT comparable with the
frozen-roster IG numbers (uniform baseline); the other six metrics are.

Two additions/deviations from the plan above, decided during implementation:

1. **Parametric blocks.** `LoraDinoV3ViT` takes two orthogonal axes:
   `lora_blocks` (which blocks get adapters; default ALL 12 — the tapped
   blocks are where features are read, but deeper outputs depend on every
   block before them, so adapting all 12 gives full freedom at trivial param
   cost) and `tap_blocks` (which blocks features are extracted from; default
   2/5/8/11, drives `out_channels`). Also parametric: `lora_targets`
   (default `('qkv', 'proj')`), `r`, `alpha`, `lora_dropout`.

2. **`forward_intermediates` instead of hooks.** The plan's claim "hooks
   capturing block outputs retain grad fine" is FALSE under gradient
   checkpointing: the block forward runs in no-grad mode (recomputed during
   backward), so hook-captured tensors are detached and LoRA would silently
   train at zero gradient. `LoraDinoV3ViT` therefore uses timm's
   `forward_intermediates()` (collects the checkpoint *outputs*, which keep
   their grad_fn), plus a `requires_grad_(True)` hook on the patch-embed
   output (reentrant-checkpoint guard, same trick as peft's
   `enable_input_require_grads`) and a first-batch non-zero-LoRA-grad
   assertion in `train_one_epoch_lora` that fails loudly if features ever
   come back detached.
