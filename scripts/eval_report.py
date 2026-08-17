"""Evaluate a checkpoint on the validation split and produce the submission's reporting assets.

Covers the mandatory reporting items (CLAUDE.md §13):
  * PSNR / SSIM per image, written to CSV for the whole val split
  * metrics reported BOTH over all pairs and over the clean subset
  * restored examples at full resolution, including BEST and WORST (failure) cases
  * a metric summary JSON

LPIPS uses the pretrained AlexNet backbone (lpips 0.1.4); pass --no-lpips to skip it.

Usage:
    python scripts/eval_report.py --checkpoint weights/unet_l1_b32_best.pt
    python scripts/eval_report.py --checkpoint weights/unet_l1_b32_best.pt --device cpu
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import PairedRestorationDataset, read_stems  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402
from src.paths import (  # noqa: E402
    CONFIGS,
    RESULTS,
    SPLIT_VAL,
    VAL_GT,
    VAL_LR,
    assert_heldout_intact,
)

SHOW = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def panel_row(axes, stem, lr, base, pred, gt, m_base, m_pred, corr):
    tag = "  [DEGENERATE: pure-noise GT]" if corr < 0.5 else ""
    panels = [
        (lr, f"NoisyLR input\n{tuple(lr.shape)}"),
        (base, f"bilinear x2\n{m_base[0]:.2f} dB / {m_base[1]:.3f}"),
        (pred, f"restored\n{m_pred[0]:.2f} dB / {m_pred[1]:.3f}"),
        (gt, f"GT\ncorr {corr:.3f}"),
    ]
    for j, (img, title) in enumerate(panels):
        axes[j].imshow(img, **SHOW)
        axes[j].set_title(f"{stem}{tag}\n{title}" if j == 0 else title, fontsize=9)
        axes[j].set_xticks([])
        axes[j].set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-show", type=int, default=4, help="best/worst cases to render")
    ap.add_argument("--no-clamp", action="store_true")
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS (needs pretrained net)")
    args = ap.parse_args()

    assert_heldout_intact()
    device = pick_device(args.device)
    clamp = not args.no_clamp

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(
        cfg.get("arch", "tiny"),
        channels=cfg.get("channels", 64),
        num_blocks=cfg.get("blocks", 8),
        base=cfg.get("base", 32),
        blocks_per_level=cfg.get("blocks_per_level", 2),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    name = args.checkpoint.stem
    out_dir = RESULTS / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {args.checkpoint}  (epoch {ck['epoch']}, "
          f"{count_parameters(model):,} params, arch {cfg.get('arch','tiny')})")
    print(f"device: {device}   clamp output to [0,1]: {clamp}\n")

    lpips_net = None
    if not args.no_lpips:
        import warnings

        import lpips as lpips_lib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lpips_net = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in lpips_net.parameters():
            p.requires_grad_(False)

    def lpips_of(pred, gt):
        """LPIPS expects 3-channel input in [-1, 1]."""
        if lpips_net is None:
            return float("nan")
        return lpips_net(
            pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1, gt.repeat(1, 3, 1, 1) * 2 - 1
        ).mean().item()

    val_stems = read_stems(SPLIT_VAL)
    degenerate = set((CONFIGS / "degenerate.txt").read_text().split())
    corr = {}
    with open(CONFIGS / "manifest.csv") as fh:
        for r in csv.DictReader(fh):
            corr[r["stem"]] = float(r["gt_lag1_corr"])

    ds = PairedRestorationDataset(VAL_GT, VAL_LR, val_stems, augment=False, cache=False)
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            lr, gt = ds[i]
            lr, gt = lr.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
            pred = model(lr)
            base = F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False)
            if clamp:
                pred, base = pred.clamp(0, 1), base.clamp(0, 1)
            rows.append({
                "stem": val_stems[i],
                "degenerate": int(val_stems[i] in degenerate),
                "gt_lag1_corr": round(corr[val_stems[i]], 6),
                "psnr": round(psnr(pred, gt).item(), 4),
                "ssim": round(ssim(pred, gt).item(), 5),
                "lpips": round(lpips_of(pred, gt), 5),
                "psnr_bilinear": round(psnr(base, gt).item(), 4),
                "ssim_bilinear": round(ssim(base, gt).item(), 5),
                "lpips_bilinear": round(lpips_of(base, gt), 5),
            })

    csv_path = out_dir / "per_image_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        s = [r for r in rows if sel(r)]
        return {
            "n": len(s),
            "psnr": float(np.mean([r["psnr"] for r in s])),
            "ssim": float(np.mean([r["ssim"] for r in s])),
            "lpips": float(np.mean([r["lpips"] for r in s])),
            "psnr_bilinear": float(np.mean([r["psnr_bilinear"] for r in s])),
            "ssim_bilinear": float(np.mean([r["ssim_bilinear"] for r in s])),
            "lpips_bilinear": float(np.mean([r["lpips_bilinear"] for r in s])),
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "arch": cfg.get("arch", "tiny"),
        "params": count_parameters(model),
        "epoch": ck["epoch"],
        "clamped": clamp,
        "all": agg(lambda r: True),
        "clean": agg(lambda r: not r["degenerate"]),
        "degenerate_only": agg(lambda r: r["degenerate"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for k in ("all", "clean", "degenerate_only"):
        a = summary[k]
        print(f"  {k:<16} n={a['n']:3d}  PSNR {a['psnr']:6.3f}  SSIM {a['ssim']:.4f}  "
              f"LPIPS {a['lpips']:.4f}   (bilinear {a['psnr_bilinear']:6.3f} / "
              f"{a['ssim_bilinear']:.4f} / {a['lpips_bilinear']:.4f})")

    # --- best / worst cases -----------------------------------------------------------------
    clean_rows = [r for r in rows if not r["degenerate"]]
    ranked = sorted(clean_rows, key=lambda r: r["psnr"] - r["psnr_bilinear"])
    worst = ranked[: args.n_show]
    best = ranked[-args.n_show:][::-1]

    for label, group in (("best", best), ("worst", worst)):
        fig, ax = plt.subplots(len(group), 4, figsize=(14, 3.6 * len(group)), squeeze=False)
        for i, r in enumerate(group):
            j = val_stems.index(r["stem"])
            lr, gt = ds[j]
            with torch.no_grad():
                lr_d = lr.unsqueeze(0).to(device)
                pred = model(lr_d)
                base = F.interpolate(lr_d, scale_factor=2, mode="bilinear", align_corners=False)
                if clamp:
                    pred, base = pred.clamp(0, 1), base.clamp(0, 1)
            panel_row(
                ax[i], r["stem"], lr[0].numpy(), base[0, 0].cpu().numpy(),
                pred[0, 0].cpu().numpy(), gt[0].numpy(),
                (r["psnr_bilinear"], r["ssim_bilinear"]), (r["psnr"], r["ssim"]),
                r["gt_lag1_corr"],
            )
        title = ("Best cases (largest gain over bilinear)" if label == "best"
                 else "Worst cases / failure analysis (smallest gain over bilinear)")
        fig.suptitle(f"{name} — {title}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_cases.png", dpi=110)
        plt.close(fig)

    print(f"\nwrote {csv_path}")
    print(f"      {out_dir/'summary.json'}, best_cases.png, worst_cases.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
