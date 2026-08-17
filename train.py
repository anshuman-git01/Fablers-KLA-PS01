"""Train a restoration model on the paired NoisyLR -> GT dataset.

Reproducible by construction: every run writes its full config, per-epoch metrics and
checkpoints under results/runs/<name>/, and all seeds are fixed. See CLAUDE.md §5 (axis 3:
training & compute hygiene) and §13 (reporting requirements).

Validation is reported TWO ways every epoch:
  * ``all``   — all 320 val pairs, matching the hidden test distribution.
  * ``clean`` — excluding the 9 degenerate pure-noise GT pairs, which are unrestorable and
                only add a constant offset (CLAUDE.md §3b).
Degenerate pairs are KEPT in training; only the reporting is split.

Usage:
    python train.py --name baseline_l1 --epochs 60
    python train.py --name wide --channels 96 --blocks 12 --epochs 80
"""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import PairedRestorationDataset, read_stems
from src.losses import LOSSES, build_loss
from src.metrics import psnr, ssim
from src.model import ARCHITECTURES, build_model, count_parameters
from src.paths import (
    CONFIGS,
    RESULTS,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TRAIN_GT,
    TRAIN_LR,
    VAL_GT,
    VAL_LR,
    WEIGHTS,
    assert_heldout_intact,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to ``device``.

    ``non_blocking=True`` is used ONLY on CUDA, where it is meaningful (async copy from pinned
    host memory). On MPS an async copy from pageable memory can race and deliver a tensor full
    of NaN — observed here as a non-finite training loss whose inputs were NaN while the model
    and the on-disk data were both clean.
    """
    return t.to(device, non_blocking=(device.type == "cuda"))


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, degenerate: set[str], stems: list[str], clamp: bool,
             lpips_net=None):
    """Return metrics over all val pairs and over the clean subset only.

    LPIPS is included when ``lpips_net`` is given — it is one third of KLA's scored metric, so
    checkpoint selection is blind without it.
    """
    model.eval()
    acc = {"all": [], "clean": []}
    idx = 0
    for lr, gt in loader:
        lr, gt = to_device(lr, device), to_device(gt, device)
        pred = model(lr)
        if clamp:
            pred = pred.clamp(0.0, 1.0)
        # per-image so we can partition by stem
        for b in range(lr.shape[0]):
            p, g = pred[b : b + 1], gt[b : b + 1]
            lp = (
                lpips_net(p.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1,
                          g.repeat(1, 3, 1, 1) * 2 - 1).mean().item()
                if lpips_net is not None else float("nan")
            )
            rec = (psnr(p, g).item(), ssim(p, g).item(), F.l1_loss(p, g).item(), lp)
            acc["all"].append(rec)
            if stems[idx] not in degenerate:
                acc["clean"].append(rec)
            idx += 1

    out = {}
    for key, recs in acc.items():
        arr = np.array(recs)
        out[f"psnr_{key}"] = float(arr[:, 0].mean())
        out[f"ssim_{key}"] = float(arr[:, 1].mean())
        out[f"l1_{key}"] = float(arr[:, 2].mean())
        out[f"lpips_{key}"] = float(arr[:, 3].mean())
        out[f"n_{key}"] = len(recs)
    return out


def selection_score(m: dict, ref: dict, criterion: str) -> float:
    """Score used to pick the 'best' checkpoint. Higher is better.

    ``combined`` (default) is the fair criterion: it measures each metric as a RELATIVE
    improvement over the bilinear reference and averages the three equally.

    Selecting on ``psnr`` alone structurally penalises perceptual-loss runs — a run that trades
    1% PSNR for a 44% LPIPS improvement is scored as strictly worse, and its best epoch is
    picked before the perceptual benefit has developed. That actually happened to
    `unet_ssimlpips_b32` (CLAUDE.md §15c).

    Equal weighting is the maximum-entropy choice given that KLA's weights are undisclosed
    (§5). It is NOT a claim that the true weights are equal.
    """
    if criterion == "psnr":
        return m["psnr_all"]
    if criterion == "ssim":
        return m["ssim_all"]
    if criterion == "lpips":
        return -m["lpips_all"]  # lower is better
    if criterion != "combined":
        raise ValueError(f"unknown selection criterion {criterion!r}")

    r_psnr = (m["psnr_all"] - ref["psnr_all"]) / abs(ref["psnr_all"])
    r_ssim = (m["ssim_all"] - ref["ssim_all"]) / abs(ref["ssim_all"])
    if np.isnan(m.get("lpips_all", float("nan"))):
        return (r_psnr + r_ssim) / 2
    # LPIPS is lower-is-better, so improvement is a reduction
    r_lpips = (ref["lpips_all"] - m["lpips_all"]) / abs(ref["lpips_all"])
    return (r_psnr + r_ssim + r_lpips) / 3


@torch.no_grad()
def bilinear_reference(loader, device, degenerate, stems, clamp, lpips_net=None):
    """Zero-effort floor: plain bilinear x2 upsampling of the noisy input."""
    class _Bilinear(torch.nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    return evaluate(_Bilinear().to(device), loader, device, degenerate, stems, clamp, lpips_net)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", type=str, required=True, help="run name")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--arch", type=str, default="tiny", choices=sorted(ARCHITECTURES))
    ap.add_argument("--channels", type=int, default=64, help="tiny: width")
    ap.add_argument("--blocks", type=int, default=8, help="tiny: depth")
    ap.add_argument("--base", type=int, default=32, help="unet: base width")
    ap.add_argument("--blocks-per-level", type=int, default=2, help="unet: res blocks per level")
    ap.add_argument("--loss", type=str, default="l1", choices=LOSSES)
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=0.15)
    ap.add_argument("--w-lpips", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-clamp-eval", action="store_true",
                    help="evaluate raw model output instead of clamping to [0,1]")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--select", type=str, default="combined",
                    choices=["combined", "psnr", "ssim", "lpips"],
                    help="checkpoint selection criterion; 'combined' weights PSNR/SSIM/LPIPS "
                         "equally as relative improvement over the bilinear reference")
    ap.add_argument("--no-val-lpips", action="store_true",
                    help="skip LPIPS during validation (faster, but selection goes blind)")
    args = ap.parse_args()

    assert_heldout_intact()
    set_seed(args.seed)
    device = pick_device(args.device)
    clamp = not args.no_clamp_eval

    run_dir = RESULTS / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = WEIGHTS / f"{args.name}_best.pt"
    last_path = WEIGHTS / f"{args.name}_last.pt"

    cfg = vars(args) | {
        "device": str(device),
        "torch": torch.__version__,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))

    # --- data ---------------------------------------------------------------------------------
    train_stems = read_stems(SPLIT_TRAIN)
    val_stems = read_stems(SPLIT_VAL)
    degenerate = set((CONFIGS / "degenerate.txt").read_text().split())

    t0 = time.time()
    train_ds = PairedRestorationDataset(
        TRAIN_GT, TRAIN_LR, train_stems,
        augment=not args.no_augment, cache=not args.no_cache,
    )
    val_ds = PairedRestorationDataset(
        VAL_GT, VAL_LR, val_stems, augment=False, cache=not args.no_cache,
    )
    print(f"\ndata: {len(train_ds)} train / {len(val_ds)} val "
          f"({len(degenerate & set(val_stems))} degenerate in val) "
          f"loaded in {time.time()-t0:.1f}s")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    val_lpips_net = None
    if not args.no_val_lpips:
        import warnings

        import lpips as lpips_lib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_lpips_net = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
        for p_ in val_lpips_net.parameters():
            p_.requires_grad_(False)

    # --- baseline reference -------------------------------------------------------------------
    ref = bilinear_reference(val_ld, device, degenerate, val_stems, clamp, val_lpips_net)
    print(f"bilinear x2 floor:  all PSNR {ref['psnr_all']:.3f}  SSIM {ref['ssim_all']:.4f}  "
          f"LPIPS {ref['lpips_all']:.4f}")
    (run_dir / "baseline_bilinear.json").write_text(json.dumps(ref, indent=2))

    # --- model --------------------------------------------------------------------------------
    model = build_model(
        args.arch,
        channels=args.channels,
        num_blocks=args.blocks,
        base=args.base,
        blocks_per_level=args.blocks_per_level,
    ).to(device)
    n_params = count_parameters(model)
    desc = (f"c{args.channels}/b{args.blocks}" if args.arch == "tiny"
            else f"base{args.base}/bpl{args.blocks_per_level}")
    print(f"model: {args.arch} {desc}, {n_params:,} params\n")

    criterion = build_loss(
        args.loss, device, w_l1=args.w_l1, w_ssim=args.w_ssim, w_lpips=args.w_lpips
    )
    print(f"loss: {args.loss} (w_l1={criterion.w_l1}, w_ssim={criterion.w_ssim}, "
          f"w_lpips={criterion.w_lpips})\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    metrics_path = run_dir / "metrics.csv"
    # train_l1 is always the plain L1 *component* so it stays comparable across every run,
    # whatever loss is being optimized; train_loss is the weighted total actually minimized.
    fields = ["epoch", "train_l1", "train_loss", "train_ssim", "train_lpips", "lr", "epoch_s",
              "psnr_all", "ssim_all", "lpips_all", "l1_all",
              "psnr_clean", "ssim_clean", "lpips_clean", "l1_clean", "score"]
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        running, nb = 0.0, 0
        acc_parts: dict[str, float] = {}
        for lr_b, gt_b in train_ld:
            lr_b = to_device(lr_b, device)
            gt_b = to_device(gt_b, device)
            opt.zero_grad(set_to_none=True)
            out = model(lr_b)
            loss, parts = criterion(out, gt_b)
            for k, v in parts.items():
                acc_parts[k] = acc_parts.get(k, 0.0) + v
            # Guard: a non-finite loss silently poisons the epoch average and every
            # subsequent weight update. Fail loudly with enough context to diagnose.
            if not torch.isfinite(loss):
                bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch} batch {nb}: {loss.item()}\n"
                    f"  input finite : {bool(torch.isfinite(lr_b).all())}\n"
                    f"  target finite: {bool(torch.isfinite(gt_b).all())}\n"
                    f"  output finite: {bool(torch.isfinite(out).all())}\n"
                    f"  input range  : [{lr_b.min():.4f}, {lr_b.max():.4f}]\n"
                    f"  non-finite params: {bad or 'none'}"
                )
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        mean_parts = {k: v / max(nb, 1) for k, v in acc_parts.items()}
        train_l1 = mean_parts.get("l1", running / max(nb, 1))
        epoch_s = time.time() - te

        row = {"epoch": epoch, "train_l1": round(train_l1, 6),
               "train_loss": round(mean_parts.get("total", train_l1), 6),
               "train_ssim": round(mean_parts["ssim"], 6) if "ssim" in mean_parts else "",
               "train_lpips": round(mean_parts["lpips"], 6) if "lpips" in mean_parts else "",
               "lr": opt.param_groups[0]["lr"], "epoch_s": round(epoch_s, 1)}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            m = evaluate(model, val_ld, device, degenerate, val_stems, clamp,
                         val_lpips_net)
            row |= {k: round(v, 6) for k, v in m.items() if not k.startswith("n_")}
            score = selection_score(m, ref, args.select)
            row["score"] = round(score, 6)
            payload = {"model": model.state_dict(), "config": cfg, "epoch": epoch,
                       "metrics": m, "n_params": n_params, "score": score,
                       "select": args.select, "bilinear_ref": ref}
            # ALWAYS keep the final epoch, independently of selection: a criterion change
            # must never make an already-trained epoch unrecoverable.
            torch.save(payload, last_path)
            flag = ""
            if score > best:
                best = score
                torch.save(payload, ckpt_path)
                flag = "  <- best"
            print(f"epoch {epoch:3d}/{args.epochs}  train_l1 {train_l1:.5f}  "
                  f"| PSNR {m['psnr_all']:6.3f}  SSIM {m['ssim_all']:.4f}  "
                  f"LPIPS {m['lpips_all']:.4f}  | score {score:+.5f}"
                  f"  | {epoch_s:5.1f}s{flag}")
        else:
            print(f"epoch {epoch:3d}/{args.epochs}  train_l1 {train_l1:.5f}  | {epoch_s:5.1f}s")

        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(row)

    # --- summary ------------------------------------------------------------------------------
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = ck["metrics"]
    print("\n" + "=" * 78)
    print(f"best epoch {ck['epoch']}  ({n_params:,} params)")
    print(f"  ALL   ({m['n_all']:3d} pairs)  PSNR {m['psnr_all']:.3f} dB  "
          f"SSIM {m['ssim_all']:.4f}   (bilinear {ref['psnr_all']:.3f} / {ref['ssim_all']:.4f}, "
          f"{m['psnr_all']-ref['psnr_all']:+.3f} dB)")
    print(f"  CLEAN ({m['n_clean']:3d} pairs)  PSNR {m['psnr_clean']:.3f} dB  "
          f"SSIM {m['ssim_clean']:.4f}   (bilinear {ref['psnr_clean']:.3f} / "
          f"{ref['ssim_clean']:.4f}, {m['psnr_clean']-ref['psnr_clean']:+.3f} dB)")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  metrics:    {metrics_path}")
    print("=" * 78)
    assert_heldout_intact()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
