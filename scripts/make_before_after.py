"""Before / after strip for the Idea Description slide.

Shows a degraded input beside our restored output at full image scale, so the result reads
instantly with no explanation. Ground truth is included so the comparison stays honest.

Candidates are chosen from the validation split by PSNR gain over bilinear, restricted to
structured images, then rendered at native resolution.

Usage:  python scripts/make_before_after.py [--stem 002506]
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
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model  # noqa: E402
from src.paths import RESULTS, SUFFIX, VAL_GT, VAL_LR, WEIGHTS  # noqa: E402

INK, INK_DIM, LIME, OURS, BASE = "#FFFFFF", "#AEB6D6", "#9EE84F", "#68A828", "#6C77C4"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="000889")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(WEIGHTS / "unet_ssimlpips_b32_best.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["arch"], base=cfg["base"],
                        blocks_per_level=cfg["blocks_per_level"])
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    lr_np = np.load(VAL_LR / f"{args.stem}{SUFFIX}")
    gt = np.load(VAL_GT / f"{args.stem}{SUFFIX}")
    lr = torch.from_numpy(lr_np)[None, None].to(device)
    with torch.no_grad():
        pred = model(lr).clamp(0, 1)[0, 0].cpu().numpy()
        # nearest upscale of the input, so "before" is shown at the same display size
        # without silently doing half the restoration job for it
        before = F.interpolate(lr, scale_factor=2, mode="nearest")[0, 0].cpu().numpy()

    m = {}
    for r in csv.DictReader(open(RESULTS / "eval" / "unet_ssimlpips_b32_best"
                                 / "per_image_metrics.csv")):
        if r["stem"] == args.stem:
            m = r

    panels = [
        (before, "Degraded input", f"{lr_np.shape[0]} x {lr_np.shape[1]}", BASE),
        (pred, "Restored", f"{float(m['psnr']):.1f} dB", OURS),
        (gt, "Ground truth", f"{gt.shape[0]} x {gt.shape[1]}", LIME),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for ax, (img, title, sub, col) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3.0)
        ax.set_title(title, fontsize=18, fontweight="bold", color=INK, pad=13)
        ax.text(0.5, -0.07, sub, transform=ax.transAxes, ha="center",
                fontsize=13.5, color=col, fontweight="bold")

    out = RESULTS / "figures" / "before_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}  stem {args.stem}  PSNR {float(m['psnr']):.2f} "
          f"(bilinear {float(m['psnr_bilinear']):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
