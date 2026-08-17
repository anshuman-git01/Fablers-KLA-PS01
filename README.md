# KLA PS01 — AI-Based Restoration of Degraded Images

Restores degraded grayscale images (speckle noise + additive Gaussian noise + 2× downsampling,
applied in an undisclosed order) back to clean, full-resolution ground truth.

**Final model:** a multi-scale U-Net restorer (1.16 M parameters) trained with a combined
L1 + SSIM + LPIPS loss — `weights/unet_ssimlpips_b32_best.pt`. Its complete specification
(architecture, loss weights, hyperparameters, seeds, data split, metrics, checkpoint hash) is
in [configs/final_model.json](configs/final_model.json). This is the checkpoint `inference.py`
loads by default, and the one that produced `results/test_outputs/`.

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bilinear ×2 (zero-effort floor) | 24.953 | 0.6215 | 0.3842 |
| Baseline CNN (L1) | 28.362 | 0.7687 | 0.3055 |
| **Final model** | **28.680** | **0.7833** | **0.1556** |

Measured on a held-out 320-pair validation split. **Throughput: 400 images in 2.41 s
end-to-end = 6.03 ms/image** (batch 32, Apple M-series GPU via MPS), of which ~0.58 s is
one-time startup and model initialisation; the steady-state cost is 4.62 ms/image. See §8 for
exactly what the timed window includes.

---

## 1. Setup

Requires **Python 3.11+**. Tested on Python 3.14.3, macOS (MPS) and CUDA.

```bash
git clone <repository-url>
cd kla-restoration
python -m pip install -r requirements.txt
```

> On systems where `python` is not on `PATH` (common with python.org installs on macOS), use
> `python3` for every command in this README. Both were verified.

`requirements.txt` pins the full stack coherently — `torchvision==0.26.0` declares
`torch==2.11.0`, so the correct torch is installed automatically.

> ⚠️ **Do not install `torchvision` or `lpips` unpinned.** Unpinned they resolve to
> `torch==2.13.0`, which changes numerical behaviour. Always install via `requirements.txt`.

**First run needs network access once.** LPIPS downloads its AlexNet backbone (233 MB) to
`~/.cache/torch/hub/checkpoints/`. This is only needed for *training* and for *metric reporting*
— **`inference.py` does not use LPIPS and runs fully offline.**

`requirements-freeze.txt` contains a complete `pip freeze` of the development machine.

---

## 2. Running inference (primary deliverable)

```bash
python inference.py --input_dir /path/to/degraded --output_dir /path/to/restored
```

That is the whole contract — no source edits, no config files, no environment variables. The
model checkpoint is resolved automatically and carries its own architecture configuration.

Useful optional flags:

```bash
python inference.py --input_dir data/NoisyLR --output_dir results/test_outputs \
    --batch-size 32 \        # default 32; raise if GPU memory allows
    --device cuda \          # default: auto-detect cuda > mps > cpu
    --report-timing          # print the per-stage timing breakdown
```

| Flag | Default | Purpose |
|---|---|---|
| `--checkpoint` | `weights/unet_ssimlpips_b32_best.pt` | model weights |
| `--batch-size` | `32` | batch size for GPU inference |
| `--device` | `auto` | `cuda` / `mps` / `cpu` |
| `--no-clamp` | off | write raw output instead of clamping to `[0,1]` |
| `--output-format` | `npy` | `npy` or `png` (see §5) |
| `--report-timing` | off | per-stage end-to-end breakdown |
| `--limit N` | all | process only the first N files (smoke testing) |

The output directory receives **only** restored `.npy` files. Timing is written to
`results/inference_timing.json`, deliberately *outside* the output directory so an evaluator
can iterate it blindly.

---

## 3. Input / output contract

| | Input | Output |
|---|---|---|
| Format | `.npy` (NumPy array) | `.npy` (NumPy array) |
| dtype | `float32` | `float32` |
| Shape | 2-D `(H, W)`, single channel | 2-D `(2H, 2W)` |
| Range | may fall **outside** `[0,1]` | clamped to `[0,1]` |
| Filename | `<stem>.npy` | **identical** `<stem>.npy` |

- **Upscale is ×2 in each dimension, derived per file.** Nothing is hardcoded to 128→256; a
  256×256 input correctly produces 512×512.
- **Inputs are never clipped.** Values outside `[0,1]` are a genuine feature of the data
  (speckle overshoot) and carry information about local noise strength, so they are fed to the
  model as-is.
- **Arbitrary input sizes are supported.** The network downsamples twice, so inputs are
  reflection-padded to a multiple of 4 and the output cropped back. Verified on 130×130,
  127×131, 100×64 and 256×256.

---

## 4. Approach

**Architecture** — `UNetRestorer` ([src/model.py](src/model.py)): a 3-level encoder–decoder
(full → ½ → ¼ resolution) with residual blocks and skip connections, a PixelShuffle ×2 output
head, and a **global residual on top of a bilinear ×2 upsample**. The output head is
zero-initialised, so the network *starts* as an exact bilinear upsampler and only learns the
correction — which converges markedly faster.

The key design driver was **receptive field**. A flat stack of 3×3 convolutions at full
resolution reaches only 21 px and cannot distinguish speckle from genuine fine texture. The
U-Net reaches **97 px** (measured empirically via input gradients) while being *faster* per
epoch, because most computation happens at reduced resolution.

**Loss** — `1.0·L1 + 0.15·(1−SSIM) + 0.10·LPIPS` ([src/losses.py](src/losses.py)). Weights are
chosen so all three terms contribute comparably (their raw magnitudes differ ~10×).

The perceptual terms address the dominant failure mode of a pure-L1 model: on dense stochastic
texture (gravel, foliage), L1's optimum is the conditional mean, so the model *blurs* rather
than commits to detail. Adding SSIM+LPIPS **cut LPIPS by 44%** (0.2778 → 0.1556) for only
−0.32 dB PSNR, and visibly restores granular texture — see
`results/eval/unet_ssimlpips_b32_best/worst_cases.png`.

**Augmentation** — the 8 dihedral transforms (flips + 90° rotations), applied identically to
input and target. These are exactly label-preserving and resample no pixels, so they do not
disturb the noise statistics the model must learn.

---

## 5. Assumptions

1. **Output format is `.npy` float32.** No official document specifies the output format. Since
   inputs are `.npy` float32 and KLA applies no clipping or renormalisation before scoring, a
   float array is the only self-consistent choice — PNG would force 8-bit quantisation. A
   `--output-format png` escape hatch exists if this assumption is wrong.
2. **Output is clamped to `[0,1]`.** Ground truth lies in `[0,1]` and KLA scores outputs exactly
   as written, so this post-processing is applied deliberately inside our pipeline. Disable with
   `--no-clamp`.
3. **Degraded inputs are not clipped on the way in** (see §3).
4. **~2% of training ground-truth images are structureless white noise** and therefore
   unrestorable. They are **kept in training** (the hidden test set may contain the same kind of
   sample) but validation metrics are reported both including and excluding them. Flagged stems
   are listed in `configs/degenerate.txt`; the detector is lag-1 pixel correlation < 0.5.
5. **The shipped data is uniformly 128×128 → 256×256.** All 3200 training pairs and all 400 test
   inputs were measured. The documentation mentions 512×512, so the model is kept fully
   convolutional and resolution-agnostic regardless.

---

## 6. Reproducing training

Data preparation (run once; creates the 90/10 split and the degenerate-pair tags):

```bash
python scripts/make_split.py        # 3200 pairs -> 2880 train / 320 val, seed 42
python scripts/tag_degenerate.py    # flags pure-noise GT images
```

Train the final model (~70 min on an Apple M-series GPU):

```bash
python train.py --name unet_ssimlpips_b32 \
    --arch unet --base 32 --blocks-per-level 2 \
    --loss l1_ssim_lpips --w-l1 1.0 --w-ssim 0.15 --w-lpips 0.10 \
    --epochs 60 --batch-size 16 --lr 5e-4 --seed 42
```

Reproduce the baseline for comparison:

```bash
python train.py --name baseline_l1_c64b8 --arch tiny --channels 64 --blocks 8 \
    --loss l1 --epochs 60 --batch-size 16 --lr 5e-4 --seed 42
```

Every run writes `results/runs/<name>/` containing `config.json`, per-epoch `metrics.csv` and
the bilinear reference, plus `weights/<name>_best.pt` **and** `weights/<name>_last.pt`.

**Checkpoint selection** defaults to `--select combined`: each of PSNR/SSIM/LPIPS is measured as
relative improvement over the bilinear reference and the three are averaged equally. Selecting
on PSNR alone structurally penalises perceptual-loss models, so it is not the default. Equal
weighting is a maximum-entropy choice, since KLA's true weights are undisclosed.

### Evaluation and reporting

```bash
python scripts/eval_report.py --checkpoint weights/unet_ssimlpips_b32_best.pt
```

Writes per-image PSNR/SSIM/LPIPS to CSV, a summary JSON, and best/worst-case figures at full
resolution, under `results/eval/<checkpoint>/`.

---

## 7. Repository layout

```
README.md              requirements.txt        requirements-freeze.txt
train.py               inference.py         <- primary deliverable
configs/               split_train.txt, split_val.txt, manifest.csv, degenerate.txt
src/                   model.py, data.py, losses.py, metrics.py, paths.py
scripts/               make_split.py, tag_degenerate.py, inspect_samples.py,
                       eval_report.py, overfit_sanity.py
weights/               *_best.pt, *_last.pt
results/               runs/ (training logs)  eval/ (reports)  test_outputs/ (restored test set)
```

> ⚠️ **Two directories are named `NoisyLR`.** `data/train/NoisyLR/` is training input;
> `data/NoisyLR/` is the **held-out test set** and is never used for training, validation or
> model selection. This is enforced in code (`src/paths.py`).

---

## 8. Hardware, runtime and timing method

| | |
|---|---|
| Training hardware | Apple M-series GPU (PyTorch MPS backend) |
| Training time | ~70 min (60 epochs) for the final model |
| Model size | 1,156,164 parameters (4.6 MB checkpoint) |
| **End-to-end runtime (400 images)** | **2.41 s wall clock = 6.03 ms/image, 166 img/s** |
| — of which one-time startup + model init | ~0.58 s (interpreter boot, imports, checkpoint load) |
| — steady-state per image | 4.62 ms/image, 217 img/s |
| Inference batch size | 32 |
| Software | Python 3.14.3, PyTorch 2.11.0 |

**Timing method — what is included.** The headline figure follows KLA's stated benchmark
definition: *script startup and model initialisation, reading input images from disk,
performing inference on the full test set, and writing output images back to disk.* A
`time.perf_counter()` timestamp is taken in `inference.py` **before `numpy` and `torch` are
imported**, so import cost and checkpoint loading fall inside the measured window rather than
outside it. Device synchronisation (`torch.cuda.synchronize` / `torch.mps.synchronize`) is
issued before each stage boundary so asynchronous GPU work cannot be attributed to the wrong
stage. The quoted 2.41 s is the median external wall clock of three consecutive runs
(`/usr/bin/time -p`), which also captures Python interpreter boot ahead of the first in-process
timestamp; the in-process figure is 2.29 s.

Both numbers are reported because they answer different questions. **6.03 ms/image** is the
correct figure for benchmarking a single batch run of 400 images, since startup is amortised
across only that set. **4.62 ms/image** is the steady-state cost and is the number that scales:
on a larger test set the one-time 0.58 s becomes negligible and throughput approaches 217 img/s.

Per-stage breakdown for a 400-image run on MPS: imports and model init 19%, forward pass 71%,
disk write 8%, disk read 1.5%, host/device transfers 1%. On an H100 the forward pass will shrink
substantially, so **startup and I/O are expected to dominate** — that is where further
optimisation should be directed, not the model.

⚠️ **Cold-cache caveat.** The first run after a reboot, or on a machine where the PyTorch
package is not in the OS page cache, spends ~2.6 s importing `torch` instead of ~0.42 s. That
pushes the end-to-end figure to roughly 4.8 s (12.0 ms/image) for 400 images. The steady-state
per-image cost is unaffected. Timing JSON for every run is written to
`results/inference_timing.json`.

**Independent verification on NVIDIA CUDA.** `inference.py` was additionally verified on a fresh
Google Colab NVIDIA CUDA instance, cloned directly from the GitHub repository and run with zero
manual edits — a separate environment from the Apple MPS machine used for development. This
confirms the device auto-detection path, the CUDA code path and the dependency pins work on
NVIDIA hardware ahead of KLA's H100 benchmarking. The runtime figures quoted above are from the
MPS development machine; no H100 measurement is claimed.

---

## 9. External resources

| Resource | Version | Licence | Link |
|---|---|---|---|
| PyTorch | 2.11.0 | BSD-3-Clause | https://pytorch.org |
| torchvision | 0.26.0 | BSD-3-Clause | https://github.com/pytorch/vision |
| LPIPS (Zhang et al., CVPR 2018) | 0.1.4 | BSD-2-Clause | https://github.com/richzhang/PerceptualSimilarity |
| AlexNet backbone (LPIPS) | torchvision pretrained | BSD-3-Clause | https://download.pytorch.org/models/alexnet-owt-7be5be79.pth |
| NumPy | 2.4.2 | BSD-3-Clause | https://numpy.org |
| Matplotlib | 3.10.8 | PSF-based | https://matplotlib.org |

**No external training datasets were used** — the model is trained solely on the provided KLA
data. The only pretrained weights are AlexNet, used inside LPIPS as a *loss and metric*; it is
not part of the restoration model and is not used at inference.

LPIPS paper: Zhang, Isola, Efros, Shechtman, Wang, *The Unreasonable Effectiveness of Deep
Features as a Perceptual Metric*, CVPR 2018.

---

## 10. Known limitations

- **Stochastic texture remains the hardest case.** Dense gravel/foliage is improved by the
  perceptual loss but still not fully resolved; these are the lowest-scoring real images.
- **Structureless ground truth is unrestorable.** On the ~2% pure-noise pairs the model reaches
  ~19 dB versus ~29 dB on real images — an information-theoretic ceiling, not a model failure.
- **The PSNR/perceptual trade-off is untuned.** `w_lpips=0.10` was not swept; a sweep may recover
  part of the 0.32 dB PSNR while retaining most of the LPIPS gain.
- **Out-of-distribution content is untested.** The hidden test set contains image types absent
  from training, and we have no OOD validation data to measure that directly.
