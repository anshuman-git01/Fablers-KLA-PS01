"""Flag degenerate 'pure noise' GT images and record the tag in configs/manifest.csv.

Some ground-truth images are structureless white noise (CLAUDE.md §3b) — there is no spatial
correlation to recover, so they are unrestorable and act as label noise. We KEEP them in
training (they may well appear in KLA's hidden test set), but validation metrics are reported
both with and without them so the headline number stays honest.

Detector: lag-1 horizontal pixel correlation. Natural images sit ~0.95+; white noise sits ~0.

Adds ``gt_lag1_corr`` and ``degenerate`` columns to configs/manifest.csv and writes the flagged
stems to configs/degenerate.txt. Locates each stem in data/train/ or data/val/ via the split
files — never touches the held-out test set.

Usage:  python scripts/tag_degenerate.py [--threshold 0.5]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import (  # noqa: E402
    CONFIGS,
    MANIFEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SUFFIX,
    TRAIN_GT,
    VAL_GT,
    assert_heldout_intact,
)


def lag1_corr(a: np.ndarray) -> float:
    """Lag-1 horizontal pixel correlation. ~0 for white noise, ~0.95+ for natural images."""
    x, y = a[:, :-1].ravel(), a[:, 1:].ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    assert_heldout_intact()

    train_stems = set(SPLIT_TRAIN.read_text().split())
    val_stems = set(SPLIT_VAL.read_text().split())

    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh))
    print(f"[1/3] scoring {len(rows)} GT images by lag-1 correlation")

    for i, r in enumerate(rows, 1):
        stem = r["stem"]
        if stem in val_stems:
            path = VAL_GT / f"{stem}{SUFFIX}"
        elif stem in train_stems:
            path = TRAIN_GT / f"{stem}{SUFFIX}"
        else:
            raise KeyError(f"{stem} is in the manifest but in neither split file")
        c = lag1_corr(np.load(path))
        r["gt_lag1_corr"] = f"{c:.6f}"
        r["degenerate"] = int(c < args.threshold)
        if i % 800 == 0:
            print(f"      ...{i}/{len(rows)}")

    fields = ["stem", "gt_h", "gt_w", "lr_h", "lr_w", "gt_lag1_corr", "degenerate"]
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[2/3] rewrote {MANIFEST.name} with gt_lag1_corr + degenerate columns")

    degen = sorted(r["stem"] for r in rows if r["degenerate"])
    (CONFIGS / "degenerate.txt").write_text("\n".join(degen) + "\n")

    d_train = sorted(s for s in degen if s in train_stems)
    d_val = sorted(s for s in degen if s in val_stems)
    corrs = np.array([float(r["gt_lag1_corr"]) for r in rows])

    print(f"[3/3] wrote configs/degenerate.txt")
    print(f"\n  threshold          : corr < {args.threshold}")
    print(f"  median corr        : {np.median(corrs):.4f}")
    print(f"  degenerate total   : {len(degen)} / {len(rows)} ({100*len(degen)/len(rows):.2f}%)")
    print(f"    in train split   : {len(d_train)} / {len(train_stems)} "
          f"({100*len(d_train)/len(train_stems):.2f}%)")
    print(f"    in val split     : {len(d_val)} / {len(val_stems)} "
          f"({100*len(d_val)/len(val_stems):.2f}%)")
    print(f"  clean val pairs    : {len(val_stems) - len(d_val)}")
    if d_val:
        print(f"  degenerate in val  : {', '.join(d_val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
