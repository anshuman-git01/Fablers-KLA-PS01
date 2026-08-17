"""Visually and numerically inspect the GT/NoisyLR pairs in data/sample/.

⚠️  This script reads ONLY data/sample/. It never lists, globs or iterates data/train/,
    data/val/ or data/NoisyLR/ — the only path constants it imports are SAMPLE_GT and SAMPLE_LR.
    Keep it that way: it exists so we can look at data without any risk of touching the
    held-out test set.

Prints per-pair statistics and writes one figure per pair plus a contact sheet to
results/inspection/.

Usage:  python scripts/inspect_samples.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never require a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import INSPECTION, SAMPLE_GT, SAMPLE_LR, SUFFIX  # noqa: E402

# Everything is rendered on a shared, fixed intensity scale so that brightness differences
# between GT and NoisyLR are real and not matplotlib auto-scaling artefacts.
VMIN, VMAX = 0.0, 1.0
IMSHOW = dict(cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="nearest")


def describe(name: str, a: np.ndarray) -> str:
    return (f"    {name:<8} shape={str(a.shape):<12} dtype={a.dtype}  "
            f"min={a.min():+.4f}  max={a.max():+.4f}  "
            f"mean={a.mean():.4f}  std={a.std():.4f}")


def pair_figure(stem: str, gt: np.ndarray, lr: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(f"{stem}   —   NoisyLR {lr.shape} → GT {gt.shape}", fontsize=13)

    ax[0, 0].imshow(lr, **IMSHOW)
    ax[0, 0].set_title(f"NoisyLR (input) {lr.shape}\nrange [{lr.min():.3f}, {lr.max():.3f}]")

    ax[0, 1].imshow(gt, **IMSHOW)
    ax[0, 1].set_title(f"GT (target) {gt.shape}\nrange [{gt.min():.3f}, {gt.max():.3f}]")

    # --- overlaid intensity histograms (the plot KLA showed in the webinar) ---
    lo = min(gt.min(), lr.min())
    hi = max(gt.max(), lr.max())
    bins = np.linspace(lo, hi, 200)
    ax[1, 0].hist(gt.ravel(), bins=bins, density=True, alpha=0.55, label="GT", color="tab:blue")
    ax[1, 0].hist(lr.ravel(), bins=bins, density=True, alpha=0.55, label="NoisyLR",
                  color="tab:orange")
    ax[1, 0].axvline(0.0, color="k", ls="--", lw=0.8)
    ax[1, 0].axvline(1.0, color="k", ls="--", lw=0.8)
    ax[1, 0].set_xlabel("pixel intensity")
    ax[1, 0].set_ylabel("density")
    ax[1, 0].set_title("Intensity histogram (dashed = [0,1] bounds)")
    ax[1, 0].legend()

    # --- zoom crop at pixel scale, so grain texture is visible ---
    # centre 64x64 of LR against the corresponding 128x128 region of GT
    ch, cw = lr.shape[0] // 2, lr.shape[1] // 2
    k = min(32, ch, cw)
    lr_crop = lr[ch - k:ch + k, cw - k:cw + k]
    gt_crop = gt[2 * (ch - k):2 * (ch + k), 2 * (cw - k):2 * (cw + k)]
    combo = np.concatenate(
        [np.kron(lr_crop, np.ones((2, 2))), gt_crop], axis=1
    )  # LR nearest-upscaled x2 so both crops cover the same field of view
    ax[1, 1].imshow(combo, **IMSHOW)
    ax[1, 1].axvline(gt_crop.shape[1] - 0.5, color="tab:red", lw=1.5)
    ax[1, 1].set_title(f"Zoom, same field of view ({2*k}x{2*k} GT px)\n"
                       f"left: NoisyLR (nearest x2)   |   right: GT")

    for a in (ax[0, 0], ax[0, 1], ax[1, 1]):
        a.set_xticks([])
        a.set_yticks([])

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    INSPECTION.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in SAMPLE_GT.glob(f"*{SUFFIX}"))
    if not stems:
        print("data/sample/GT is empty — run scripts/make_sample.py first.")
        return 1
    print(f"Inspecting {len(stems)} pairs from data/sample/ (this is the ONLY dir read)\n")

    rows = []
    for stem in stems:
        gt = np.load(SAMPLE_GT / f"{stem}{SUFFIX}")
        lr = np.load(SAMPLE_LR / f"{stem}{SUFFIX}")

        below = float((lr < 0.0).mean())
        above = float((lr > 1.0).mean())
        scale = gt.shape[0] / lr.shape[0]

        print(f"{stem}")
        print(describe("GT", gt))
        print(describe("NoisyLR", lr))
        print(f"    scale factor        : {scale:g}x")
        print(f"    LR pixels < 0       : {below*100:.3f}%")
        print(f"    LR pixels > 1       : {above*100:.3f}%")
        print(f"    mean ratio LR/GT    : {lr.mean()/gt.mean():.4f}   "
              f"(≈1 => noise is mean-preserving)")
        print(f"    std ratio  LR/GT    : {lr.std()/gt.std():.4f}\n")

        pair_figure(stem, gt, lr, INSPECTION / f"{stem}.png")
        rows.append((stem, gt, lr))

    # --- contact sheet -------------------------------------------------------------------------
    n = len(rows)
    fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 5.2))
    ax = np.atleast_2d(ax)
    for j, (stem, gt, lr) in enumerate(rows):
        ax[0, j].imshow(lr, **IMSHOW)
        ax[0, j].set_title(stem, fontsize=8)
        ax[1, j].imshow(gt, **IMSHOW)
        for i in (0, 1):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    ax[0, 0].set_ylabel("NoisyLR", fontsize=9)
    ax[1, 0].set_ylabel("GT", fontsize=9)
    fig.suptitle("data/sample contact sheet — top: NoisyLR input, bottom: GT target")
    fig.tight_layout()
    fig.savefig(INSPECTION / "contact_sheet.png", dpi=130)
    plt.close(fig)

    # --- aggregate summary ---------------------------------------------------------------------
    all_lr = np.concatenate([lr.ravel() for _, _, lr in rows])
    all_gt = np.concatenate([gt.ravel() for _, gt, _ in rows])
    print("=" * 72)
    print("AGGREGATE over the sample")
    print(f"  GT      range [{all_gt.min():+.4f}, {all_gt.max():+.4f}]  "
          f"mean {all_gt.mean():.4f}  std {all_gt.std():.4f}")
    print(f"  NoisyLR range [{all_lr.min():+.4f}, {all_lr.max():+.4f}]  "
          f"mean {all_lr.mean():.4f}  std {all_lr.std():.4f}")
    print(f"  NoisyLR outside [0,1]: {((all_lr < 0) | (all_lr > 1)).mean()*100:.3f}% "
          f"({(all_lr < 0).mean()*100:.3f}% below, {(all_lr > 1).mean()*100:.3f}% above)")
    print(f"\nFigures written to {INSPECTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
