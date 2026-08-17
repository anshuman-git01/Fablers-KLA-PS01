"""Standalone inference: restore every degraded .npy in a directory.

    python inference.py --input_dir <dir of degraded .npy> --output_dir <dir for restored .npy>

Runs as-is with no source edits: the model checkpoint, architecture and all hyperparameters are
resolved automatically (the checkpoint stores its own config). Only the two directory arguments
are required.

Contract
--------
Input   : ``.npy``, float32, 2-D ``(H, W)``, values may fall outside [0,1] (speckle overshoot).
Output  : ``.npy``, float32, 2-D ``(2H, 2W)``, **same filename as the input**.
Scale   : x2 in each dimension, derived per file — nothing is hardcoded to 128 -> 256.

Post-processing: outputs are clamped to [0,1] to match the ground-truth range. KLA applies no
clipping or renormalisation before scoring, so this is done here deliberately and is reported.
Disable with --no-clamp.

Timing: end-to-end wall clock is the headline number, matching how KLA benchmarks (disk read ->
preprocess -> host-to-device -> forward -> device-to-host -> postprocess -> disk write). Pass
--report-timing for the per-stage breakdown.
"""

from __future__ import annotations

import time

# Captured before numpy/torch are imported, so the reported end-to-end figure includes
# import cost. KLA's benchmark definition covers "script startup and model initialization,
# reading input images from disk, performing inference on the full test set, writing output
# images back to disk", so those must sit inside the timed window.
_T_SCRIPT_START = time.perf_counter()

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_T_IMPORTS_DONE = time.perf_counter()

# Resolve the repository root from this file so the script works from any working directory.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from src.model import build_model
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        f"Could not import the model definition from {_ROOT/'src'}.\n"
        "Run inference.py from inside the repository (src/ must sit next to it)."
    ) from e

DEFAULT_CHECKPOINT = _ROOT / "weights" / "unet_ssimlpips_b32_best.pt"
SUFFIX = ".npy"


class Stopwatch:
    """Accumulates per-stage timings."""

    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    def add(self, stage: str, dt: float) -> None:
        self.t[stage] = self.t.get(stage, 0.0) + dt

    def report(self, n: int, total: float) -> str:
        lines = [f"{'stage':<22}{'seconds':>10}{'% total':>10}{'ms/image':>11}"]
        lines.append("-" * 53)
        for k, v in self.t.items():
            lines.append(f"{k:<22}{v:>10.3f}{100*v/total:>9.1f}%{1000*v/n:>11.3f}")
        lines.append("-" * 53)
        lines.append(f"{'END-TO-END WALL':<22}{total:>10.3f}{100.0:>9.1f}%{1000*total/n:>11.3f}")
        return "\n".join(lines)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = build_model(
        cfg.get("arch", "unet"),
        channels=cfg.get("channels", 64),
        num_blocks=cfg.get("blocks", 8),
        base=cfg.get("base", 32),
        blocks_per_level=cfg.get("blocks_per_level", 2),
    )
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    return model, ck, cfg


def pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, int, int]:
    """Reflection-pad H,W up to a multiple of ``m``.

    The U-Net downsamples twice, so inputs must be divisible by 4. The shipped data is 128x128
    and needs no padding, but an arbitrary OOD size must not crash the evaluator's run.
    """
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, ph, pw


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input_dir", type=Path, required=True,
                    help="directory of degraded .npy files")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="directory to write restored .npy files")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no-clamp", action="store_true",
                    help="write raw model output instead of clamping to [0,1]")
    ap.add_argument("--output-format", choices=["npy", "png"], default="npy",
                    help="npy (default, lossless float32) or png (8-bit, quantizes and clips)")
    ap.add_argument("--report-timing", action="store_true",
                    help="print the per-stage timing breakdown")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files")
    ap.add_argument("--timing-json", type=Path, default=None,
                    help="where to write the timing JSON (default: results/inference_timing.json). "
                         "Deliberately NOT inside output_dir, which must contain only outputs.")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    clamp = not args.no_clamp
    sw = Stopwatch()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    sw.add("0a imports", _T_IMPORTS_DONE - _T_SCRIPT_START)
    _t = time.perf_counter()
    model, ck, cfg = load_model(args.checkpoint, device)
    sw.add("0b model init", time.perf_counter() - _t)

    files = sorted(args.input_dir.glob(f"*{SUFFIX}"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no {SUFFIX} files found in {args.input_dir}")

    print(f"model      : {cfg.get('arch','unet')} "
          f"({sum(p.numel() for p in model.parameters()):,} params, epoch {ck.get('epoch','?')})")
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}   batch size: {args.batch_size}   clamp: {clamp}")
    print(f"input      : {len(files)} files from {args.input_dir}")
    print(f"output     : {args.output_dir}  (format: {args.output_format})\n")

    # Group by shape so every batch is rectangular. The shipped data is uniformly 128x128, so
    # this is a single group in practice, but mixed sizes must not break batching.
    groups: dict[tuple[int, int], list[Path]] = {}
    t0 = time.perf_counter()
    for f in files:
        shape = np.load(f, mmap_mode="r").shape
        groups.setdefault(tuple(shape), []).append(f)
    sw.add("scan/index", time.perf_counter() - t0)

    wall0 = time.perf_counter()
    n_done = 0
    with torch.no_grad():
        for shape, group in groups.items():
            for i in range(0, len(group), args.batch_size):
                chunk = group[i : i + args.batch_size]

                t = time.perf_counter()
                arrays = [np.load(f) for f in chunk]
                sw.add("1 disk read", time.perf_counter() - t)

                t = time.perf_counter()
                batch = torch.from_numpy(
                    np.ascontiguousarray(np.stack(arrays), dtype=np.float32)
                ).unsqueeze(1)
                batch, ph, pw = pad_to_multiple(batch, 4)
                sw.add("2 preprocess", time.perf_counter() - t)

                t = time.perf_counter()
                batch = batch.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                sw.add("3 host->device", time.perf_counter() - t)

                t = time.perf_counter()
                out = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                sw.add("4 forward", time.perf_counter() - t)

                t = time.perf_counter()
                out = out.cpu()
                sw.add("5 device->host", time.perf_counter() - t)

                t = time.perf_counter()
                if ph or pw:  # crop away the x2-scaled padding
                    out = out[..., : out.shape[-2] - 2 * ph, : out.shape[-1] - 2 * pw]
                if clamp:
                    out = out.clamp(0.0, 1.0)
                np_out = out.squeeze(1).numpy().astype(np.float32)
                sw.add("6 postprocess", time.perf_counter() - t)

                t = time.perf_counter()
                for f, arr in zip(chunk, np_out):
                    if args.output_format == "npy":
                        np.save(args.output_dir / f.name, arr)
                    else:
                        import matplotlib.image as mpimg

                        mpimg.imsave(
                            args.output_dir / f"{f.stem}.png",
                            np.clip(arr, 0, 1), cmap="gray", vmin=0, vmax=1,
                        )
                sw.add("7 disk write", time.perf_counter() - t)
                n_done += len(chunk)

    wall = time.perf_counter() - wall0
    pipeline = wall + sw.t.get("scan/index", 0.0)
    # KLA's definition: startup + model init + read + inference + write
    total = time.perf_counter() - _T_SCRIPT_START
    startup = total - pipeline

    print(f"restored {n_done} images")
    print(f"  END-TO-END (KLA definition) {total:7.3f} s   "
          f"{1000*total/n_done:6.2f} ms/image   {n_done/total:6.1f} img/s")
    print(f"  one-time startup + init     {startup:7.3f} s   "
          f"(interpreter boot before the first timestamp is not counted here)")
    print(f"  steady-state pipeline       {pipeline:7.3f} s   "
          f"{1000*pipeline/n_done:6.2f} ms/image   {n_done/pipeline:6.1f} img/s")
    if args.report_timing:
        print()
        print(sw.report(n_done, total))

    timing_path = args.timing_json or (_ROOT / "results" / "inference_timing.json")
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps({
        "n_images": n_done,
        "end_to_end_seconds": total,
        "ms_per_image": 1000 * total / n_done,
        "images_per_second": n_done / total,
        "startup_and_model_init_seconds": startup,
        "steady_state_pipeline_seconds": pipeline,
        "steady_state_ms_per_image": 1000 * pipeline / n_done,
        "device": str(device),
        "batch_size": args.batch_size,
        "checkpoint": str(args.checkpoint),
        "clamped": clamp,
        "stages": sw.t,
        "torch": torch.__version__,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
