"""Pipeline sanity check: overfit a couple of pairs and confirm the loss collapses.

This is the check KLA explicitly recommended before any real training (CLAUDE.md §15). It does
not produce a useful model — it proves the dataloader, model, loss and optimizer are wired up
correctly. If a network cannot memorize two images, something is broken.

Reads only data/sample/. Deliberately skips degenerate pure-noise GT images (CLAUDE.md §3b),
since "can it memorize white noise" is a much weaker signal than "can it memorize structure".

Pass criterion: final PSNR must clear --min-psnr AND beat the bilinear-upsample baseline by a
wide margin. Exits non-zero on failure so it can be used as a smoke test.

Usage:  python scripts/overfit_sanity.py [--steps 2000] [--n-pairs 2]
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import PairedRestorationDataset  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402
from src.model import TinyRestorer, count_parameters  # noqa: E402
from src.paths import RESULTS, SAMPLE_GT, SAMPLE_LR, SUFFIX  # noqa: E402

STRUCTURE_MIN_CORR = 0.5  # lag-1 correlation below this => degenerate noise image


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lag1_corr(a: np.ndarray) -> float:
    return float(np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1])


def structured_stems(n: int) -> list[str]:
    """Pick n sample stems whose GT has real spatial structure."""
    keep = []
    for p in sorted(SAMPLE_GT.glob(f"*{SUFFIX}")):
        c = lag1_corr(np.load(p))
        if c >= STRUCTURE_MIN_CORR:
            keep.append(p.stem)
        else:
            print(f"    skipping {p.stem}: lag-1 corr {c:.3f} -> degenerate noise GT")
        if len(keep) == n:
            break
    if len(keep) < n:
        raise RuntimeError(f"only found {len(keep)} structured samples, wanted {n}")
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-pairs", type=int, default=2)
    # 5e-4 is the safe default: 2e-3 is stable for the ~38k-param model but stalls a
    # 1.2M-param one near the bilinear baseline (see CLAUDE.md §17).
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-psnr", type=float, default=35.0)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"device: {device}   torch {torch.__version__}\n")

    print("[1/4] selecting sample pairs")
    stems = structured_stems(args.n_pairs)
    ds = PairedRestorationDataset(SAMPLE_GT, SAMPLE_LR, stems=stems)
    lr = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    gt = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
    print(f"      using {stems}")
    print(f"      lr batch {tuple(lr.shape)}  ->  gt batch {tuple(gt.shape)}")

    # zero-effort reference: what does plain bilinear upsampling score?
    with torch.no_grad():
        base = F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False)
        base_psnr = psnr(base, gt).item()
        base_ssim = ssim(base, gt).item()
    print(f"      bilinear baseline: PSNR {base_psnr:.2f} dB   SSIM {base_ssim:.4f}")

    print("\n[2/4] building model")
    model = TinyRestorer(channels=args.channels, num_blocks=args.blocks).to(device)
    print(f"      TinyRestorer, {count_parameters(model):,} trainable parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"\n[3/4] overfitting {len(ds)} pair(s) for {args.steps} steps (L1 loss)")

    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(lr)
        loss = F.l1_loss(pred, gt)
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                p = model(lr)
                cur_psnr = psnr(p, gt).item()
                cur_ssim = ssim(p, gt).item()
            history.append((step, loss.item(), cur_psnr))
            print(f"      step {step:5d}   L1 {loss.item():.6f}   "
                  f"PSNR {cur_psnr:6.2f} dB   SSIM {cur_ssim:.4f}")

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(lr)
        final_psnr = psnr(pred, gt).item()
        final_ssim = ssim(pred, gt).item()
        final_l1 = F.l1_loss(pred, gt).item()

    print(f"\n      {elapsed:.1f}s total, {1000*elapsed/args.steps:.1f} ms/step")

    # --- figure ---------------------------------------------------------------------------------
    print("\n[4/4] writing figure")
    out_png = RESULTS / "sanity_overfit.png"
    n = len(ds)
    fig, ax = plt.subplots(n, 4, figsize=(15, 4 * n), squeeze=False)
    show = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for i in range(n):
        panels = [
            (lr[i, 0].cpu(), f"NoisyLR input\n{tuple(lr.shape[2:])}"),
            (base[i, 0].cpu(), f"bilinear x2\n{base_psnr:.2f} dB"),
            (pred[i, 0].detach().cpu(), f"overfit prediction\n{final_psnr:.2f} dB"),
            (gt[i, 0].cpu(), f"GT target\n{tuple(gt.shape[2:])}"),
        ]
        for j, (img, title) in enumerate(panels):
            ax[i, j].imshow(img, **show)
            ax[i, j].set_title(f"{stems[i]}  —  {title}" if j == 0 else title, fontsize=10)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    fig.suptitle("Pipeline sanity check — overfitting a tiny model on a couple of pairs")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"      {out_png}")

    # --- verdict --------------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"final: L1 {final_l1:.6f}   PSNR {final_psnr:.2f} dB   SSIM {final_ssim:.4f}")
    print(f"gain over bilinear baseline: {final_psnr - base_psnr:+.2f} dB")
    ok = final_psnr >= args.min_psnr and final_psnr > base_psnr + 5.0
    if ok:
        print(f"PASS — model memorizes the pairs, so the pipeline is wired correctly.")
        print("NOTE: this number is meaningless as a quality result; it is a plumbing test.")
    else:
        print(f"FAIL — expected PSNR >= {args.min_psnr} dB and >5 dB over baseline.")
        print("Something in the data/model/loss path is likely broken.")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
