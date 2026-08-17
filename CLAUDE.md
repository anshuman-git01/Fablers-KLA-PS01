# KLA PS01 — AI-Based Restoration of Degraded Images

**This file is the single source of truth for the project.** Everything in `docs/` has been
distilled here. Do not re-read `docs/description.md`, `docs/KLA_help_document.pdf`,
`docs/Submission_Requirements.md`, or `docs/webinar_transcript.md` — if something is missing
here, add it here.

Facts marked **[VERIFIED]** were confirmed by reading the data on disk and outrank the docs.
Facts marked **[UNCONFIRMED]** are open — see §16.

---

## 1. Task

Given a degraded grayscale image (noisy + low-resolution), produce a restored image matching a
clean, full-resolution ground truth. Training data is paired: `NoisyLR` input ↔ `GT` target.
The held-out test set gives only degraded inputs; KLA keeps the ground truth for scoring.
Formally: GT `x` → degraded `y = F(x)`; learn `F⁻¹`. This is image restoration
(joint denoising + super-resolution), not defect detection.

## 2. Degradations — exactly three, nothing else

| # | Degradation | Nature | Notes |
|---|---|---|---|
| 1 | **Speckle noise** | **Multiplicative** | Scales with local image intensity — bright regions get amplified noise. This is what pushes values outside `[0,1]`. |
| 2 | **Additive Gaussian noise** | **Additive**, zero-mean | σ sampled from a range, not fixed per image. Looks **grainy**. |
| 3 | **Downsampling** | Resolution loss | **Exactly 2×** in each dimension. 512→256 or 256→128. |

- **The order of application is undisclosed and may be any permutation.** Do not infer order
  from the order they are listed in any document — KLA said this explicitly, twice.
- The model does **not** need to identify the order. One-shot restoration is fine; a staged
  approach is also fine. Detecting the order is optional and may or may not help.
- **Blur is NOT one of the three degradations.** It appears in the webinars only as a generic
  teaching example of image degradation. It is not in the benchmark. Do not model it as a
  separate stage. (See §16 open question 2.)
- KLA suggested reading the *physics* of these noise models as a hint toward the true pipeline.

## 3. Data contract **[VERIFIED on disk]**

| Property | Value |
|---|---|
| File format | **`.npy`** NumPy arrays — **not** PNG/TIFF/JPEG |
| dtype | `float32` |
| Array shape | **2-D `(H, W)`** — no channel axis |
| Filenames | 6-digit zero-padded stems: `000000.npy`, `000001.npy`, … |
| Pairing | GT and NoisyLR stems match exactly, 1:1 |
| Grayscale | Single channel. Colour is explicitly out of scope. |
| Scale factor | NoisyLR is **exactly half** GT in each dimension |

### 3a. Resolution is a single fixed scale **[VERIFIED — contradicts the docs]**

Measured over all 3200 training pairs (`configs/manifest.csv`) **and** all 400 held-out test
inputs:

| Set | Input shape | Target shape | Count |
|---|---|---|---|
| `data/train/` + `data/val/` | `(128, 128)` | `(256, 256)` | 3200 (**100%**) |
| `data/NoisyLR/` (held-out test) | `(128, 128)` | `(256, 256)` implied | 400 (**100%**) |

**There are zero 512×512 pairs.** Every document claims GT is "512×512 or 256×256"; the shipped
data has only 256×256 GT. The task is a single fixed scale: **128×128 → 256×256, ×2**.

Consequences:
- Fixed-size architectures are viable, and batching is trivial — every tensor is the same shape.
  No padding, no bucketing, no tiling.
- **Still keep the model fully convolutional / resolution-agnostic anyway.** The docs repeatedly
  promise 512×512 may appear, and the OOD half of the hidden test set is the plausible place for
  it. Costs nothing to preserve; avoids a catastrophic failure if a 256×256 input shows up.
- `inference.py` must not hardcode 128→256. Derive output size as `2 × input` per file.

**Value ranges:**
- **GT is normalized to `[0,1]`.**
- **NoisyLR may fall outside `[0,1]`** — on both tails. Verified example: `[-0.0026, 1.3258]`.
- This is a **feature of the dataset, not a bug.** KLA said explicitly: *think about how to use
  this information rather than just clipping it to `[0,1]`.* The out-of-range magnitude carries
  signal about local speckle intensity. **Do not blindly clip the input.**

## 3b. Measured degradation characteristics **[VERIFIED]**

From `scripts/inspect_samples.py` over `data/sample/` (8 pairs) plus a full-split scan.

**Image content: natural photographs.** Rock faces, clouds/sky, mountain landscape, rock strata,
wooden fence, soft textures. **Not semiconductor imagery.** Settles §16 Q1.

**The degradation is visibly grainy**, high-frequency, pixel-level — not soft or hazy. Settles
§16 Q2. There is no evidence of a separate blur kernel beyond downsampling itself.

**Noise is mean-preserving but variance-inflating:**

| Quantity | Observed range across samples |
|---|---|
| `mean(LR) / mean(GT)` | **1.0000 – 1.0046** — essentially exactly 1 |
| `std(LR) / std(GT)` | **0.93 – 1.41** — varies a lot per image |
| LR pixels outside `[0,1]` | 0.85% aggregate (0.25% below 0, 0.60% above 1) |
| LR min / max seen | −0.1305 / +1.5024 |

Interpretation: consistent with zero-mean additive Gaussian plus unit-mean multiplicative
speckle — both preserve the mean while adding variance. **The per-image std ratio varying from
0.93 to 1.41 confirms noise severity is sampled per image, not fixed** (as KLA stated). A model
conditioned on, or robust to, varying noise level matters more than tuning for one σ.

The out-of-range fraction is **asymmetric** (more above 1 than below 0) and correlates with image
brightness — the signature of intensity-dependent multiplicative speckle. This asymmetry is
usable signal: the size of the overshoot is informative about local speckle strength.

### ⚠️ ~1% of GT images are pure noise (degenerate pairs)

Lag-1 horizontal pixel correlation over all 2880 training GT images:

| Correlation | Count | Share |
|---|---|---|
| < 0.1 (**pure white noise, unrestorable**) | 29 | 1.01% |
| < 0.5 | 53 | 1.84% |
| < 0.7 | 103 | 3.58% |
| median | 0.9650 | — |

Some GT images are genuinely structureless white noise (e.g. `000405`, `002981`, `002982`,
`000353`). For these there is **nothing to restore** — no spatial correlation exists to recover,
so the best achievable output is a conditional mean and PSNR is capped near the noise floor.

Implications to decide later (not yet acted on):
- They act as **label noise**; a few dozen such pairs can drag training and, worse, distort
  validation metrics if they land in the val split.
- Options: exclude below a correlation threshold, down-weight them, or keep them and report
  metrics both with and without. **Do not silently drop them** — whatever is chosen must be
  stated in the submission, since KLA's hidden test set may contain the same kind of sample.
- Re-run the scan with the snippet in the git history or recompute via lag-1 correlation on
  `data/train/GT`.

## 4. Folder map — ⚠️ THE TWO-`NoisyLR` TRAP

There are two different directories named `NoisyLR`. Confusing them silently destroys the
validity of every result.

```
data/
  train/GT/          TRAINING targets      (2880 after split)
  train/NoisyLR/     TRAINING inputs       (2880 after split)   ← paired with train/GT
  val/GT/            VALIDATION targets    (320)
  val/NoisyLR/       VALIDATION inputs     (320)
  sample/GT/         8 pairs, inspection only
  sample/NoisyLR/    8 pairs, inspection only
  NoisyLR/           ⛔ HELD-OUT TEST SET  (400, no GT exists)
```

**Hard rule:** nothing under `data/NoisyLR/` may ever enter a training or validation split, be
used for model selection, or be used to compute a reported metric. It is for **final inference
only**. In code it is exposed as `HELDOUT_TEST_LR` in `src/paths.py` and may only be referenced
by `inference.py`. Originally 3200 train pairs + 400 test files.

## 5. Evaluation — three axes

1. **Restoration quality.** A **fixed weighted combination of PSNR + SSIM + LPIPS**, computed
   internally by KLA against hidden GT. The exact weights are **not disclosed**. Reported as one
   number, not per-metric. Covers both in-distribution and out-of-distribution content.
2. **End-to-end throughput** on a common **NVIDIA H100**. This is **not** just the forward pass —
   it includes disk read → preprocess → host-to-device transfer → model execution →
   device-to-host transfer → postprocess → disk write. Optimize the whole pipeline.
3. **Training & compute hygiene.** Reproducibility, clean sequential experiment design,
   environment specification, code quality, efficient data pipeline, standard ML/DL practice.

Final ranking combines all three. **No target score and no latency threshold were given** —
deliberately. Any of the three axes can decide the finalists.

## 6. ⚠️ Hard scoring rule — no clipping on their side

> KLA applies **no clipping and no renormalization** before scoring. Outputs are scored
> **exactly as our pipeline saves them.**

Any clipping, denormalization or range correction must happen **inside our solution**, as part
of the model or an explicit post-processing step. A model emitting out-of-range or unnormalized
values will be scored on those raw values.

## 7. Test-set behaviour

- Mix of **in-distribution** (content similar to training) and **out-of-distribution** content.
  The proportion and the relative scoring weight are **not disclosed**.
- OOD shift is in **image content only** — different subject matter/sources/structures.
- **The degradation mechanisms are identical.** No unseen degradation types.
- Noise **severity** is sampled from a similar range to training — "may be slightly different,
  but not drastically". We do **not** need to generalize to unseen severities or unseen
  combinations; that is explicitly out of scope.
- Test images are ~256×256 or 512×512 at GT scale, so inputs are ~128×128 or ~256×256.
  Not "too large" — tiling is unlikely to be needed.
- Evaluation is on the **full-resolution image only**; no separate zoomed-in detail scoring.

## 8. What is allowed / required

**Allowed:**
- Any architecture: CNN, transformer, algorithm-unrolling, other published architecture, or a
  justified custom/hybrid design. KLA recommends reusing published architectures over inventing
  one, since they come with ablations and known generalization behaviour.
- **Public pretrained weights** (HuggingFace, torch.hub, timm, TF Model Zoo, …) as initialization.
- **Public external datasets** under a permissive licence that allows competition use.
- **Generating extra synthetic degraded pairs from the provided GT images** using our own
  degradation pipeline. Explicitly permitted and encouraged.
- Frequency-domain methods — allowed, not mandatory, and called out as a reasonable direction.
- CUDA-specific optimizations (it will run on an NVIDIA GPU).
- No parameter-count limit — but oversized models cost throughput, which is separately scored.

**Required disclosure:** every external dataset and model must be listed with **name, link,
licence, and paper or model/dataset card**.

## 9. Inference script contract — the most important deliverable

> *"The evaluation script is the most important file in your repository. It will be used AS-IS
> by KLA's benchmarking team. If your script does not run without manual edits, your submission
> cannot be benchmarked, and unscored submissions cannot win."*

- Standalone **`.py`** — **not** a Jupyter notebook.
- Accepts **two arguments: an input directory and an output directory**.
- Loads **every** degraded file in the input dir, restores it, writes each output to the output dir.
- Must run with **zero manual source-code edits** — no hardcoded local paths, no notebook cells.
- Preserves required file naming and format (see §16 open question 4).
- Runs on an NVIDIA GPU. **Batch processing strongly preferred** when GPU memory permits.
- Ships with all weights, config and dependencies needed to execute.
- Must be dry-run in a clean environment before submitting.

## 10. Mandated repository layout

From the KLA PDF:

```
README.md   requirements.txt   train.py   inference.py
configs/    src/    weights/    results/    solution_presentation.pptx
```

Our working tree adds `scripts/` (one-off data prep), `data/` and `docs/`.

## 11. Phase-1 deliverables (all mandatory)

1. **Solution PPT/PPTX** — problem understanding, approach, model, losses, augmentation,
   experiments, PSNR/SSIM/LPIPS, runtime, examples, limitations, external resources, next steps.
2. **GitHub repository link** — must be **public** and accessible, with clear folder structure.
3. **Inference script** — per §9.
4. **Training code** — reproduces the submitted checkpoint. Script or notebook.
5. **Model weights + config** — final checkpoint and architecture/config files. Use Git LFS or a
   Drive/HuggingFace link if large.
6. **README.md** — exact environment setup, training and inference commands, input/output
   contract, assumptions. A reviewer must clone and run inference **without contacting us**.
7. **requirements.txt** — complete `pip freeze` from the training environment.
8. **Restored test outputs** — a folder of the actual restored images our model produced on the
   test set, plus a metric summary and failure analysis.

## 12. PPT format

The **organizer's Idea Submission Template is binding** (from `Submission_Requirements.md`);
the KLA PDF's 12-slide list is a *recommendation* to fold into it. See §16 open question 6.

- **Max 8–9 slides.** Remove the template's instruction slide. **Save as PDF.**
- Filename: **`TeamName_KLA_PS01.pdf`** (e.g. `VisionForge_KLA_PS01.pdf`).
- Binding slide order: 1 Team details · 2 Problem statement (select "AI-Based Restoration of
  Degraded Images") · 3 Idea description · 4 Proposed solution (+ pipeline diagram) ·
  5 Innovation & uniqueness · 6 Results (SSIM/PSNR/LPIPS + before/after) · 7 Technology &
  feasibility (stack, GPU, training time, model size, inference time) · 8 GitHub + video link ·
  9 References.
- Topics from the PDF's 12-slide version to fold in: dataset analysis & degradation
  observations, preprocessing/augmentation, experiment tracking & baseline comparison, runtime
  and batch size, failure cases and limitations, external-resource disclosure.

## 13. Validation & reporting requirements

- Create a validation split **not used for training or model-selection leakage**.
- Report **PSNR, SSIM and LPIPS**, plus any additional metric used to select the final model.
- **Compare at least one baseline** against the final method.
- Show restored examples at **full image resolution**, including **both successful and failed**
  cases. At least one failure case is mandatory.
- Report **end-to-end runtime, batch size, hardware, software versions and timing method**.
- Track experiments, **random seeds**, hyperparameters, checkpoints and the final configuration.

## 14. Deadline

**Phase 1: 19 August 2026** — confirmed by the user. This **supersedes the 16 August 2026** date
printed in `KLA_help_document.pdf` §5, which is stale. Do not re-derive the deadline from the PDF.

Later stages: ~15 teams per problem statement shortlisted for round 2 (additional task or demo),
then 5 teams per problem statement to the grand finale. Shortlisted submissions get run on the
hidden test data on a shared H100. **Do not retrain on hidden test inputs.**

## 15. Guidance worth keeping (from the KLA sessions)

- **Sanity-check first:** build the smallest end-to-end pipeline (loader → tiny model → simple
  loss) and **overfit 1–2 pairs**. If it can't overfit two images, there's a bug. Do this before
  any real training run.
- **Look at the output images, not just the metrics.** The single most common mistake KLA cites.
  Aggregate metrics hide "the model is missing speckle" or "it's smoothing away detail".
- **Change one thing at a time and track every experiment.** Otherwise it becomes a jumble.
- **Augmentation is the most under-explored lever for OOD robustness** — KLA's own words, and
  their top recommendation for the generalization axis.
- **Don't blur to remove noise** — it destroys the detail the super-resolution half must recover.
- **Don't introduce ringing or artificial patterns** when sharpening.
- Data pipeline speed matters for *training* too, not just deployment — time should be spent in
  forward/backward, not blocked on disk I/O.
- Losses to consider: L1, L2/MSE, SSIM-as-loss, perceptual/LPIPS, frequency-domain. Pick based
  on which degradation each term counters, and justify the choice.
- Reasonable compute is sufficient — KLA sized the dataset so free Colab/Kaggle GPUs can crack it.

## 15a. Pipeline sanity check — PASSED (recorded baseline)

`scripts/overfit_sanity.py` — overfits a few `data/sample/` pairs to prove the
dataloader → model → loss → optimizer path is correct. Not a quality result.

Setup: `TinyRestorer` (residual on top of a bilinear ×2 upsample, PixelShuffle head, zero-init
residual head so it starts *as* a bilinear upsampler), L1 loss, Adam, seed 42, MPS.

| Config | Params | LR | Steps | Final L1 | PSNR | vs bilinear | Verdict |
|---|---|---|---|---|---|---|---|
| c32/b4, 2 pairs | 38k | 2e-3 | 2000 | 0.00974 | 37.55 dB | +9.65 dB | pass |
| c32/b4, 2 pairs (**default**) | 38k | 5e-4 | 2000 | 0.01083 | 36.75 dB | +8.85 dB | pass |
| c128/b8, 1 pair | 1.19M | 2e-3 | 4000 | 0.02888 | 28.09 dB | **+0.03 dB** | **FAIL** |
| c128/b8, 1 pair | 1.19M | 1e-4 | 3000 | 0.00612 | 41.58 dB | +13.52 dB | pass |
| c128/b8, 1 pair | 1.19M | 5e-4 | 3000 | **0.00185** | **50.89 dB** | +22.83 dB | pass |

**Reference number: bilinear ×2 upsampling of the noisy input scores ~27.9–28.1 dB PSNR /
~0.71–0.74 SSIM.** That is the zero-effort floor every real model must beat, and it is the
cheapest "baseline" the submission requires.

### Findings worth keeping

1. **Learning rate is the live failure mode, not capacity.** At lr=2e-3 the 1.2M-param model
   sat at the bilinear baseline (+0.03 dB) for 4000 steps — it looked exactly like a broken
   pipeline but was just an unstable LR. The identical model at 5e-4 reaches 50.9 dB.
   **Default is now 5e-4; scale LR down as the model grows, and never diagnose a stalled run
   as a bug before sweeping LR.**
2. **The small model plateaus for a legitimate reason.** 38k params cannot memorize 2×256×256 =
   131k target pixels, so ~37 dB is a capacity ceiling. Driving loss to near-zero required more
   capacity — which is what confirms the pipeline is sound.
3. **Zero-init the residual head.** The model starts as an exact bilinear upsampler, so step 1
   already scores ~27.6 dB instead of garbage. Materially faster convergence.
4. Throughput on this Mac: ~6 ms/step at 38k params, ~47 ms/step at 1.2M (batch 1–2, MPS).

## 15c. Recorded baseline — `baseline_l1_c64b8`

First real training run. **This is the baseline the final method must beat** (required by
§13/§11). Artefacts: `results/runs/baseline_l1_c64b8/`, `weights/baseline_l1_c64b8_best.pt`.

Config: `TinyRestorer` c64/b8, 298,372 params · L1 loss · Adam lr 5e-4 · cosine to 0 ·
batch 16 · dihedral augmentation · 60 epochs · seed 42 · MPS · 54.2 min total (~54 s/epoch).

| Metric | Bilinear ×2 floor | Baseline (best, epoch 58) | Gain |
|---|---|---|---|
| PSNR **all** (320 pairs) | 24.953 dB | **28.362 dB** | **+3.408 dB** |
| SSIM **all** | 0.6215 | **0.7687** | +0.147 |
| PSNR **clean** (311 pairs) | 25.139 dB | **28.634 dB** | **+3.494 dB** |
| SSIM **clean** | 0.6287 | **0.7814** | +0.153 |

**Reporting note:** the `clean` subset scores ~0.27 dB above `all`. The 9 degenerate pairs act
as a near-constant drag, not a ranking distortion — so either number ranks experiments the same
way. Quote **`all`** as the headline (it matches the hidden-test distribution) and `clean`
alongside it.

### Experiment log

Single-variable comparisons, all at L1 · Adam lr 5e-4 · cosine→0 · batch 16 · dihedral aug ·
60 epochs · seed 42 · MPS. Headline metric is **PSNR all**.

All three scored metrics, measured by `scripts/eval_report.py` on the 320-pair val split
(clamped to [0,1]). **LPIPS: lower is better.**

| Run | Loss | Params | PSNR all | SSIM all | **LPIPS all** | s/epoch |
|---|---|---|---|---|---|---|
| — bilinear ×2 floor | — | 0 | 24.953 | 0.6215 | 0.3842 | — |
| `baseline_l1_c64b8` | L1 | 298k | 28.362 | 0.7687 | 0.3055 | 54 |
| `unet_l1_b32` | L1 | 1.16M | 29.002 | 0.7870 | 0.2778 | 58 |
| `unet_l1_b48` | L1 | 2.60M | **29.083** | **0.7894** | 0.2751 | 131 |
| `unet_ssimlpips_b32` | **L1+SSIM+LPIPS** | 1.16M | 28.680 | 0.7833 | **0.1556** | ~70 |

Clean-subset equivalents: `unet_l1_b32` 29.283 / 0.7987 / 0.2725 · `unet_l1_b48` 29.364 /
0.8009 / 0.2700 · `unet_ssimlpips_b32` 28.955 / 0.7943 / 0.1527.

### ⭐ The combined loss is a large perceptual win for a small fidelity cost

Versus `unet_l1_b32`, the combined loss gives:

| Metric | Change | Relative |
|---|---|---|
| LPIPS | **0.2778 → 0.1556** | **−44%** (large improvement) |
| PSNR | 29.002 → 28.680 | −0.32 dB (−1.1%) |
| SSIM | 0.7870 → 0.7833 | −0.0037 (−0.5%) |

Measured as improvement over the bilinear floor, the combined loss delivers **2.2× the LPIPS
gain** (−0.229 vs −0.106) for **8% less PSNR gain** and **2% less SSIM gain**.

**The mechanism hypothesis was confirmed visually.** In
`results/eval/unet_ssimlpips_b32_best/worst_cases.png`, image `000818` (gravel) now retains
granular pebble texture that the L1 model averaged into a smooth wash. The perceptual term does
exactly what it was added to do: stop L1 regressing to the conditional mean on stochastic texture.

**Which model wins depends on KLA's undisclosed metric weights (§5).** A −44% LPIPS move for
−1.1% PSNR is a favourable trade under any weighting that gives LPIPS non-trivial weight, and
LPIPS is explicitly one of the three components. Do not dismiss this run on PSNR alone.

### ✅ Checkpoint selection fixed (was structurally biased)

The original criterion was `psnr_all` alone, which **systematically penalised perceptual-loss
runs**. Under it, `unet_ssimlpips_b32` ranked 3rd of 4 despite the best LPIPS by a wide margin,
and its "best" epoch was picked at 39 while validation LPIPS was still improving through 60.

`train.py` now defaults to `--select combined`: each metric is measured as **relative improvement
over the bilinear reference**, and the three are averaged equally. Equal weighting is the
maximum-entropy choice given KLA's undisclosed weights (§5) — not a claim they are equal.

Re-ranking the existing checkpoints under it:

| Run | PSNR | SSIM | LPIPS | combined | rank | (rank by PSNR) |
|---|---|---|---|---|---|---|
| **`unet_ssimlpips_b32`** | 28.680 | 0.7833 | **0.1556** | **0.3349** | **#1** | #3 |
| `unet_l1_b48` | 29.083 | 0.7894 | 0.2751 | 0.2398 | #2 | #1 |
| `unet_l1_b32` | 29.002 | 0.7870 | 0.2778 | 0.2352 | #3 | #2 |
| `baseline_l1_c64b8` | 28.362 | 0.7687 | 0.3055 | 0.1927 | #4 | #4 |

Two further fixes, both to prevent losing work:
- **`weights/<name>_last.pt` is now always saved** every epoch, independently of selection. A
  criterion change must never make an already-trained epoch unrecoverable. This bit us:
  `unet_ssimlpips_b32`'s epoch-60 weights **no longer exist** and cannot be evaluated.
- **Validation now computes LPIPS every epoch** (`--no-val-lpips` to disable), so selection is
  never blind to a third of the scored metric again.

### 🔒 FINAL MODEL — LOCKED 16 Aug 2026

**`weights/unet_ssimlpips_b32_best.pt`** — this is the submitted model. Decision made by the
user; the `w_lpips` sweep was explicitly skipped.

| | |
|---|---|
| Architecture | `unet` base32 / blocks-per-level 2, ×2 scale, RF 97 px |
| Parameters | 1,156,164 (4.6 MB) |
| Loss | `l1_ssim_lpips` — 1.0·L1 + 0.15·(1−SSIM) + 0.10·LPIPS |
| Epoch | 39 of 60 |
| SHA-256 | `8f5a6ea9cd09b48e8c25f25480b40860156a0124a72eb8e7b68d1763d5f81398` |
| Val (all, 320) | **PSNR 28.680 · SSIM 0.7833 · LPIPS 0.1556** |
| Val (clean, 311) | PSNR 28.955 · SSIM 0.7943 · LPIPS 0.1527 |
| Throughput | 4.64 ms/image end-to-end (400 imgs, batch 32, MPS) |

Full spec: **`configs/final_model.json`** (architecture, loss weights, training hyperparameters,
seeds, data split, post-processing, metrics, checkpoint hash).

It is already the default checkpoint in `inference.py`, and `results/test_outputs/` (400 files)
was produced by it. **Do not silently swap the final model** — any change means regenerating
`results/test_outputs/`, `configs/final_model.json` and the README results table.

Known caveat, recorded honestly: this run predates the `--select combined` fix, so its epoch was
chosen by `psnr_all`. It still ranks #1 of 4 under the corrected criterion, but its own epoch-60
weights were never saved and cannot be recovered (see the selection-fix note above).

### Indicated next step: tune the LPIPS weight

`w_lpips=0.10` may overshoot. A sweep (0.03 / 0.05) should trace the PSNR-vs-LPIPS knee and may
recover most of the 0.32 dB while keeping the bulk of the LPIPS gain.

**U-Net vs baseline: +0.640 dB / +0.018 SSIM.** Total over the bilinear floor: **+4.049 dB**.

The receptive-field hypothesis was correct in direction but the gain is moderate, so it was not
the only thing limiting the baseline.

### Width does not pay — capacity is ruled out

`unet_l1_b48` (2.6M params, 2.2× b32) gained only **+0.081 dB / +0.002 SSIM** for **2.3× the
epoch time** (131 s vs 58 s). At matched epochs the margin was a flat +0.04 to +0.09 dB the whole
way and never widened.

Combined with validation L1 sitting *below* training L1 in every run, this rules out capacity as
the binding constraint. It is also the wrong direction for the separately-scored throughput axis:
2.3× slower for +0.08 dB would lose more on throughput than it gains on quality.

**Use `base32` as the working architecture.** Do not pursue further width increases without a
new reason.

### Diagnosis: underfitting, architecture-limited

`train_l1` converged to 0.03145 and was flat for the last ~15 epochs with LR annealed to 0.
Validation tracked training the whole way — **no overfitting gap at all**. The model is not
data-limited or regularization-limited; it is capacity/receptive-field limited.

Cause: `TinyRestorer` is a flat stack of 3×3 convs at full resolution. Receptive field is only
~17 px and there is no multi-scale context, so it cannot distinguish speckle from genuine
fine texture — exactly the discrimination this task needs. More epochs will not help; more
width alone probably will not either.

Next architectural step should add multi-scale context (encoder–decoder with skips) and/or
start from pretrained restoration weights, which KLA explicitly permits and recommends.

### Still underfitting after the U-Net — confirmed, not inferred

`unet_l1_b32` converged to `train_l1` 0.02945 (baseline was 0.03145) with LR annealed to 0 and
was flat for the last ~10 epochs. Critically, **validation L1 (0.02894) is *lower* than training
L1 (0.02945)** — training sees dihedral augmentation and the 53 degenerate pairs, validation
sees neither. There is no overfitting gap whatsoever, in either run.

So the constraint is still capacity/optimization, not data or regularization. Indicated next
levers, in order of expected value:
1. **More capacity** — wider U-Net (`--base 48/64`) and/or `--blocks-per-level 3`.
2. **Longer schedule** — both runs annealed to LR 0 at exactly 60 epochs; that may be premature
   for a larger model.
3. **Loss** — L1 alone optimizes pixel fidelity only. SSIM and LPIPS are two thirds of the
   scored metric, and neither is being optimized for at all yet.

Do **not** reach for regularization or more augmentation to fix this: there is no overfitting
to regularize away.

### Failure analysis — the model over-smooths stochastic texture

From `results/eval/unet_l1_b32_best/worst_cases.png`. The worst cases are not random: **all of
them are dense high-frequency stochastic texture** — gravel, foliage, dense vegetation, rock
detritus. Their GT lag-1 correlations (0.553, 0.768, 0.819, 0.875) are the lowest among the
non-degenerate images, i.e. these are precisely the real images whose texture is statistically
closest to noise.

The restored outputs look plausible but visibly **smoother than GT**: the fine granular detail
is averaged away. Metrics agree — on these images the model gains only ~+0.4–0.6 dB over
bilinear versus +4.1 dB on the clean average.

**Mechanism:** this is L1 regression-to-the-mean. Where the model cannot decide whether
high-frequency content is signal or noise, the L1-optimal prediction is the conditional median —
i.e. blur it. KLA warned about exactly this: *"Do not blur the image to remove noise, that
destroys useful information."*

**Implication:** this is a *loss* problem, not a capacity problem, and it is the single clearest
lever available. L1 optimizes pixel fidelity alone, yet SSIM and LPIPS are two thirds of the
scored metric and both punish exactly this kind of texture loss. Candidate next experiments:
L1 + SSIM loss, a frequency-domain (FFT) term, or a light perceptual term.

There is a real tension to manage: sharpening raises SSIM/LPIPS but can *lower* PSNR, and the
hidden metric weighting is undisclosed (§5). Report all three metrics for any loss change rather
than optimizing PSNR alone.

## 15e. `inference.py` — built and verified

```bash
python inference.py --input_dir <degraded .npy dir> --output_dir <restored .npy dir>
```

Zero source edits required; the checkpoint carries its own architecture config. Defaults to
`weights/unet_ssimlpips_b32_best.pt`.

**Verified on the real held-out test set (400 files):**

| Check | Result |
|---|---|
| Output count / filenames | 400, identical stems to input |
| dtype / ndim | `float32`, 2-D |
| Scale | exactly ×2 per file, derived — not hardcoded |
| Value range | `[0.0000, 1.0000]`, all finite |
| Runs from any cwd | yes (paths resolved from `__file__`) |
| **Reproduces eval metrics** | **PSNR 28.680 · SSIM 0.7833 · LPIPS 0.1556 — exact match** |

That last row matters: running `inference.py` over `data/val/NoisyLR` and scoring the written
`.npy` files against val GT reproduces `eval_report.py` to 3 dp on all three metrics, proving
the inference path has no preprocessing drift from the evaluated path.

**Throughput (400 images, MPS, batch 32): 1.86 s end-to-end = 4.64 ms/image, 215 img/s.**
The forward pass is 84% of wall time; disk write 10%, read 2%. On an H100 the forward will
shrink sharply, so **I/O will likely dominate there** — that is where further optimization
should go, not the model.

Robustness (all verified, since the docs promise sizes the shipped data never contains):
reflection-padding to a multiple of 4 handles non-divisible and non-square inputs —
130×130, 127×131, 100×64 and 256×256→512×512 all produce exactly 2× output.

Two deliberate choices, both reported rather than hidden:
- **Output is clamped to [0,1]** (`--no-clamp` disables). KLA applies no clipping (§6), so this
  is our explicit post-processing step. GT is in [0,1], so it can only help.
- **The timing JSON is written to `results/`, never into `output_dir`.** The output directory
  must contain only restored `.npy` files, since the evaluator may iterate it blindly.

`--output-format {npy,png}` hedges §16 Q4; default `npy`, which is the only format coherent with
"no clipping or renormalisation before scoring".

Final test outputs (deliverable #8) are in `results/test_outputs/` — 400 restored `.npy` files.

## 15f. Packaging status

`README.md` and `requirements.txt` are written and verified. **Every command in the README was
executed verbatim and works**, including `--report-timing`, `--no-clamp`, `--output-format png`,
`--device cpu` and `--limit`.

**`requirements.txt` no longer needs the `--no-deps` dance.** `torchvision==0.26.0` declares
`torch==2.11.0`, so pinning the stack forces the correct torch: `pip install -r requirements.txt`
resolves cleanly with torch unchanged (verified via `pip install --dry-run`). The earlier
torch-2.13 upgrade risk came *only* from installing unpinned. `requirements-freeze.txt` holds the
complete 95-package `pip freeze` for the record.

Verified facts that the README depends on:
- **`inference.py` never imports lpips** — confirmed `lpips` is absent from `sys.modules` after
  importing it. So inference runs fully offline; only training and metric reporting need the
  233 MB AlexNet download.
- `python` is **not** on PATH on this machine (python.org install provides only `python3`). The
  README calls this out explicitly so a reviewer does not fail on the first command.

Phase-1 deliverable status: PPT ❌ · repo ✅ · inference script ✅ · training code ✅ ·
weights+config ✅ · README ✅ · requirements ✅ · restored test outputs ✅ (400 files in
`results/test_outputs/`).

**Remaining: the solution PPT (§12), and the deferred `w_lpips` sweep.**

## 15d. Environment — pinned versions matter

| Package | Version | Note |
|---|---|---|
| torch | **2.11.0** | **Do not upgrade.** All recorded results use this. |
| torchvision | 0.26.0 | installed `--no-deps`; the pairing for torch 2.11 |
| lpips | 0.1.4 | installed `--no-deps` |
| numpy / scipy / matplotlib | 2.4.2 / 1.17.1 / 3.10.8 | |

⚠️ **`pip install torchvision lpips` resolves to torch 2.13.0 and would silently upgrade torch.**
Both packages were therefore installed with `--no-deps` against the existing torch 2.11.0, and
the torch version was verified unchanged afterwards. Use the same approach when rebuilding the
environment, and pin all of this in `requirements.txt`.

LPIPS AlexNet weights are cached at `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
(233 MB, downloaded from download.pytorch.org). A clean-machine setup needs network access for
this on first run — note it in the README, since evaluators must be able to run from scratch.

## 15b. Gotchas hit during development — do not re-learn these

1. **`non_blocking=True` corrupts tensors on MPS.** Symptom: `train_l1 nan` while the on-disk
   data was verified clean, model params stayed finite, and val PSNR sat *exactly* on the
   bilinear baseline. The NaN guard showed the **input** tensor arriving as NaN — an async copy
   from pageable host memory racing on MPS. It only reproduced once the baseline eval had
   already allocated MPS memory, which is why isolated repros missed it.
   **Fix:** `to_device()` in `train.py` enables `non_blocking` only on CUDA, where it is
   meaningful (pinned host memory). Do not reintroduce it for MPS.
2. **`train.py` has a non-finite-loss guard.** It raises immediately with input/target/output
   finiteness, input range and any non-finite params. Keep it — a NaN loss otherwise poisons the
   epoch average and every later weight update while still *looking* like a plausible run.
3. **A stalled run is an LR problem before it is a bug.** See §15a: a 1.2M-param model sat at
   +0.03 dB over baseline for 4000 steps purely because of lr=2e-3. Sweep LR before debugging.
4. **Degenerate pure-noise pairs are kept in training, excluded only from `clean` reporting.**
   `configs/degenerate.txt` holds the 62 flagged stems (53 train / 9 val). Decision made by the
   user; do not silently drop them.

## 16. Open questions — UNCONFIRMED, do not treat as settled

1. ~~**Semiconductor vs natural images.**~~ **RESOLVED [VERIFIED] — natural images.** Visual
   inspection of `data/sample/` shows rock faces, clouds/sky, a mountain landscape, rock strata,
   and a wooden fence. No semiconductor structures whatsoever. `description.md`'s "semiconductor
   structures" and the PDF's title are **wrong about the shipped data**; Webinar 1's "normal
   natural images" is correct. → A natural-image pretrained restoration backbone is a sound
   initialization.
2. ~~**How Gaussian noise looks.**~~ **RESOLVED [VERIFIED] — grainy, not hazy.** The degradation
   is visibly high-frequency pixel grain. `description.md`'s "soft and hazy, edges lose
   sharpness" row is an **error** describing blur. No evidence of a separate blur kernel beyond
   what downsampling itself causes. → Do not model blur.
3. ~~**Resolution mix.**~~ **RESOLVED [VERIFIED]** — see §3a. There is no mix: everything is
   128×128 → 256×256. The docs' "512×512 or 256×256" does not match the shipped data.
4. **Output file format — highest-risk unknown.** No doc states it; the PDF defers to
   "official dataset/evaluator instructions" we don't have. Inputs are `.npy` float32 2-D, and
   §6 says KLA does not clip or renormalize — which is only coherent if outputs are float
   arrays, since PNG forces 8-bit quantization and clipping. **Working assumption: write `.npy`,
   `float32`, 2-D `(H,W)`, same stem as the input file.** Hedge: give `inference.py` an
   `--output-format {npy,png}` flag defaulting to `npy`. Worth confirming via the portal.
5. **"Two types of signal loss" vs three degradations.** `description.md`'s opening sentence says
   two, its own table lists three. **Three is correct**, confirmed everywhere else. Settled.
6. **Two conflicting PPT specs.** Organizer's 9-slide template vs the PDF's 12-slide structure.
   **Resolution: organizer template is binding**, fold the PDF's topics in. See §12. Settled.
