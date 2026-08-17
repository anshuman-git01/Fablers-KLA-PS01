"""Side-by-side crop showing what the perceptual loss actually changed.

Renders the same validation image restored by two models that differ ONLY in their loss:
    unet_l1_b32          L1 only
    unet_ssimlpips_b32   L1 + SSIM + LPIPS
alongside the ground truth, zoomed to pixel scale so the texture difference is visible.

Default image is 000818 (gravel), where PSNR is effectively tied between the two models but
LPIPS differs by 27%. That makes the point precisely: the gain is perceptual, not a fidelity
trade, and it is invisible to PSNR alone.

Usage:  python scripts/make_loss_comparison.py [--stem 000818] [--crop 96]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model  # noqa: E402
from src.paths import RESULTS, SUFFIX, VAL_GT, VAL_LR, WEIGHTS  # noqa: E402

INK, INK_DIM, LIME, OURS = "#FFFFFF", "#AEB6D6", "#9EE84F", "#68A828"
BASE = "#6C77C4"


def load(ckpt: Path, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = build_model(cfg.get("arch", "unet"), channels=cfg.get("channels", 64),
                    num_blocks=cfg.get("blocks", 8), base=cfg.get("base", 32),
                    blocks_per_level=cfg.get("blocks_per_level", 2))
    m.load_state_dict(ck["model"])
    return m.eval().to(device)


def per_image(run: str, stem: str) -> dict:
    for r in csv.DictReader(open(RESULTS / "eval" / run / "per_image_metrics.csv")):
        if r["stem"] == stem:
            return r
    return {}


def best_crop(gt: np.ndarray, size: int) -> tuple[int, int]:
    """Pick the window with the most high-frequency energy, so the zoom shows texture."""
    best, pos = -1.0, (0, 0)
    step = 16
    for y in range(0, gt.shape[0] - size + 1, step):
        for x in range(0, gt.shape[1] - size + 1, step):
            w = gt[y:y + size, x:x + size]
            e = float(np.abs(np.diff(w, axis=0)).mean() + np.abs(np.diff(w, axis=1)).mean())
            if e > best:
                best, pos = e, (y, x)
    return pos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="000818")
    ap.add_argument("--crop", type=int, default=96, help="crop size in GT pixels")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lr = torch.from_numpy(np.load(VAL_LR / f"{args.stem}{SUFFIX}"))[None, None].to(device)
    gt = np.load(VAL_GT / f"{args.stem}{SUFFIX}")

    with torch.no_grad():
        pred_l1 = load(WEIGHTS / "unet_l1_b32_best.pt", device)(lr).clamp(0, 1)[0, 0].cpu().numpy()
        pred_ours = load(WEIGHTS / "unet_ssimlpips_b32_best.pt", device)(lr).clamp(0, 1)[0, 0].cpu().numpy()

    y, x = best_crop(gt, args.crop)
    s = args.crop
    m_l1 = per_image("unet_l1_b32_best", args.stem)
    m_ours = per_image("unet_ssimlpips_b32_best", args.stem)

    panels = [
        (pred_l1[y:y + s, x:x + s], "L1 loss only",
         f"LPIPS {float(m_l1['lpips']):.3f}", BASE),
        (pred_ours[y:y + s, x:x + s], "L1 + SSIM + LPIPS",
         f"LPIPS {float(m_ours['lpips']):.3f}", OURS),
        (gt[y:y + s, x:x + s], "Ground truth", "reference", LIME),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for ax, (img, title, sub, col) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3.0)
        ax.set_title(title, fontsize=18, fontweight="bold", color=INK, pad=14)
        ax.text(0.5, -0.075, sub, transform=ax.transAxes, ha="center",
                fontsize=13.5, color=col, fontweight="bold")

    fig.suptitle("Same image. Same model size. Only the loss changed.",
                 fontsize=17, color=LIME, style="italic", y=1.02)
    # Footer sits clear of the per-panel LPIPS labels (which are at -0.075 in axes
    # coords); tight_layout reserves the strip below for it.
    fig.text(0.5, 0.012,
             f"PSNR is effectively tied ({float(m_l1['psnr']):.2f} vs "
             f"{float(m_ours['psnr']):.2f} dB). The perceptual loss keeps the grain "
             "instead of averaging it away.",
             ha="center", fontsize=13, color=INK_DIM)

    out = RESULTS / "figures" / "loss_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}  (crop {s}x{s} at y={y} x={x})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
