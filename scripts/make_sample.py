"""Copy a handful of GT+NoisyLR pairs into data/sample/ for visual inspection.

Draws only from configs/split_train.txt, so the sample can never contain validation data and
certainly not the held-out test set. Copies rather than moves, so data/train/ stays complete.

Stratified across GT resolution classes when more than one exists (currently there is only one:
256x256), otherwise spread evenly across the stem range so the sample isn't all consecutive
files from one part of the dataset.

Usage:  python scripts/make_sample.py [-n 8] [--seed 42]
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import (  # noqa: E402
    MANIFEST,
    SAMPLE_GT,
    SAMPLE_LR,
    SPLIT_TRAIN,
    SUFFIX,
    TRAIN_GT,
    TRAIN_LR,
    assert_heldout_intact,
    assert_not_heldout,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=8, help="number of pairs to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert_not_heldout(TRAIN_GT, TRAIN_LR, SAMPLE_GT, SAMPLE_LR)
    assert_heldout_intact()

    train_stems = set(SPLIT_TRAIN.read_text().split())
    with open(MANIFEST) as fh:
        rows = [r for r in csv.DictReader(fh) if r["stem"] in train_stems]
    print(f"[1/3] {len(rows)} candidate stems from configs/split_train.txt")

    # group by GT resolution
    strata: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        strata.setdefault((r["gt_h"], r["gt_w"]), []).append(r["stem"])

    picks: list[str] = []
    per = max(1, args.n // len(strata))
    for key in sorted(strata):
        members = sorted(strata[key])
        take = min(per, len(members))
        # even stride across the stem range rather than a random clump
        step = len(members) / take
        picks += [members[int(i * step)] for i in range(take)]
    picks = sorted(dict.fromkeys(picks))[: args.n]

    print(f"[2/3] selected {len(picks)} stems across "
          f"{len(strata)} resolution class(es): {', '.join(picks)}")

    for d in (SAMPLE_GT, SAMPLE_LR):
        d.mkdir(parents=True, exist_ok=True)
    for stem in picks:
        fn = f"{stem}{SUFFIX}"
        shutil.copy2(TRAIN_GT / fn, SAMPLE_GT / fn)
        shutil.copy2(TRAIN_LR / fn, SAMPLE_LR / fn)
        assert (SAMPLE_GT / fn).exists() and (SAMPLE_LR / fn).exists()
        # copy, not move: originals must survive
        assert (TRAIN_GT / fn).exists() and (TRAIN_LR / fn).exists()

    assert_heldout_intact()
    print(f"[3/3] copied {len(picks)} pairs into data/sample/ (originals left in data/train/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
