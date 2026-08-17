"""Render the submission demo video: results/demo_video.mp4

Motion-graphics style, not a slideshow. Every segment animates: eased count-ups, camera
push-ins built from continuously interpolated crop rectangles, and wipe reveals that turn the
degraded input into the restored output in place.

Frames are generated with PIL and piped to ffmpeg as raw RGB, so nothing touches disk between
render and encode.

Structure (30 fps):
    opening        3.0 s   title animates in
    problem        4.0 s   camera push-in on a noisy region
    restoration   7 x 3 s  wipe reveal + GT panel + eased metric count-ups
    loss compare   5.0 s   push-in with cross-dissolve between the two losses
    headline       5.0 s   four staggered eased count-ups
    closing        2.5 s   team + repo

Usage:  python scripts/make_demo_video.py [--fps 30] [--out results/demo_video.mp4]
"""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model  # noqa: E402
from src.paths import RESULTS, SUFFIX, VAL_GT, VAL_LR, WEIGHTS  # noqa: E402

W, H = 1920, 1080
BG = (13, 19, 56)
LIME = (158, 232, 79)
INK = (255, 255, 255)
DIM = (174, 182, 214)
BLUE = (108, 119, 196)
GREEN = (104, 168, 40)

FONT_DIR = Path(
    "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/"
    "site-packages/matplotlib/mpl-data/fonts/ttf"
)
_font_cache: dict = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _font_cache:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        _font_cache[key] = ImageFont.truetype(str(FONT_DIR / name), size)
    return _font_cache[key]


# ------------------------------------------------------------------ easing ---
def clamp01(t: float) -> float:
    return max(0.0, min(1.0, t))


def ease_out(t: float) -> float:
    """Cubic ease-out. Fast start, gentle settle."""
    return 1 - (1 - clamp01(t)) ** 3


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ------------------------------------------------------------------ drawing ---
def new_frame() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def text(img, xy, s, size, color=INK, bold=False, anchor="mm", alpha=1.0,
         spacing_px=0):
    """Draw text with optional uniform alpha (composited, so fades are smooth)."""
    if alpha <= 0.001:
        return
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(size, bold)
    if spacing_px:
        # manual letter-spacing for the title
        total = sum(d.textlength(ch, font=f) + spacing_px for ch in s) - spacing_px
        x = xy[0] - total / 2 if anchor[0] == "m" else xy[0]
        for ch in s:
            d.text((x, xy[1]), ch, font=f, fill=color + (int(255 * clamp01(alpha)),),
                   anchor="l" + anchor[1])
            x += d.textlength(ch, font=f) + spacing_px
    else:
        d.text(xy, s, font=f, fill=color + (int(255 * clamp01(alpha)),), anchor=anchor)
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))


def to_rgb(arr: np.ndarray) -> Image.Image:
    a = np.clip(arr, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")


def crop_zoom(arr: np.ndarray, cx: float, cy: float, frac: float, out: int) -> Image.Image:
    """Camera-style push-in: crop a sub-rectangle then scale it to `out` px.

    `frac` is the crop size as a fraction of the image (1.0 = full frame). Because frac and
    the centre are interpolated continuously between frames, the motion reads as a smooth
    dolly rather than discrete steps.
    """
    h, w = arr.shape
    size = max(8.0, min(h, w) * frac)
    x0 = clamp01((cx - size / 2 / w)) * w
    y0 = clamp01((cy - size / 2 / h)) * h
    x0 = min(x0, w - size)
    y0 = min(y0, h - size)
    img = to_rgb(arr)
    box = (x0, y0, x0 + size, y0 + size)
    return img.resize((out, out), Image.LANCZOS, box=box)


def panel(img, im: Image.Image, cx: int, cy: int, border=None, bw=4):
    x, y = cx - im.width // 2, cy - im.height // 2
    if border:
        ImageDraw.Draw(img).rectangle(
            [x - bw, y - bw, x + im.width + bw - 1, y + im.height + bw - 1], fill=border)
    img.paste(im, (x, y))


def wipe(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    """Reveal b over a with a hard vertical boundary and a bright leading edge."""
    out = a.copy()
    x = int(a.width * clamp01(t))
    if x > 0:
        out.paste(b.crop((0, 0, x, b.height)), (0, 0))
    if 0 < x < a.width:
        d = ImageDraw.Draw(out)
        d.rectangle([x - 3, 0, x + 1, a.height], fill=LIME)
    return out


def dissolve(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    return Image.blend(a, b, clamp01(t))


def progress_bar(img, y, t, width=900, color=LIME):
    x0 = (W - width) // 2
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y, x0 + width, y + 4], fill=(40, 52, 105))
    d.rectangle([x0, y, x0 + int(width * clamp01(t)), y + 4], fill=color)


# ------------------------------------------------------------------- assets ---
def load_assets(n_samples: int):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def load_model(name):
        ck = torch.load(WEIGHTS / name, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        m = build_model(cfg["arch"], base=cfg.get("base", 32),
                        blocks_per_level=cfg.get("blocks_per_level", 2),
                        channels=cfg.get("channels", 64), num_blocks=cfg.get("blocks", 8))
        m.load_state_dict(ck["model"])
        return m.eval().to(device)

    ours = load_model("unet_ssimlpips_b32_best.pt")
    l1 = load_model("unet_l1_b32_best.pt")

    rows = {r["stem"]: r for r in csv.DictReader(
        open(RESULTS / "eval" / "unet_ssimlpips_b32_best" / "per_image_metrics.csv"))}
    # structured images with the largest visible gain, spread across the split
    cand = [r for r in rows.values()
            if not int(r["degenerate"]) and float(r["gt_lag1_corr"]) > 0.80]
    cand.sort(key=lambda r: -(float(r["psnr"]) - float(r["psnr_bilinear"])))
    picks = cand[:n_samples]

    samples = []
    with torch.no_grad():
        for r in picks:
            stem = r["stem"]
            lr = np.load(VAL_LR / f"{stem}{SUFFIX}")
            gt = np.load(VAL_GT / f"{stem}{SUFFIX}")
            t = torch.from_numpy(lr)[None, None].to(device)
            pred = ours(t).clamp(0, 1)[0, 0].cpu().numpy()
            samples.append({
                "stem": stem, "lr": lr, "gt": gt, "pred": pred,
                "psnr": float(r["psnr"]), "ssim": float(r["ssim"]),
            })

        g = "000818"  # gravel: PSNR tied, LPIPS far apart
        glr = torch.from_numpy(np.load(VAL_LR / f"{g}{SUFFIX}"))[None, None].to(device)
        gravel = {
            "l1": l1(glr).clamp(0, 1)[0, 0].cpu().numpy(),
            "ours": ours(glr).clamp(0, 1)[0, 0].cpu().numpy(),
        }
    return samples, gravel


# ----------------------------------------------------------------- segments ---
def seg_opening(fps):
    n = int(3.0 * fps)
    for i in range(n):
        t = i / fps
        f = new_frame()
        # title rises and fades in with letter-spacing
        a1 = ease_out(t / 0.9)
        dy = lerp(40, 0, a1)
        text(f, (W // 2, int(H * 0.40 + dy)), "FABLERS", 128, INK, True,
             alpha=a1, spacing_px=lerp(28, 10, a1))
        # rule expands from centre
        a2 = ease_in_out((t - 0.7) / 0.7)
        if a2 > 0:
            wr = int(520 * a2)
            ImageDraw.Draw(f).rectangle(
                [W // 2 - wr, int(H * 0.50), W // 2 + wr, int(H * 0.50) + 4], fill=LIME)
        # subtitle types in
        a3 = clamp01((t - 1.15) / 1.25)
        s = "AI-Based Restoration of Degraded Images"
        k = int(len(s) * ease_out(a3))
        if k:
            text(f, (W // 2, int(H * 0.585)), s[:k], 52, DIM, False, alpha=1.0)
        a4 = clamp01((t - 2.3) / 0.6)
        text(f, (W // 2, int(H * 0.70)), "KLA  ·  PS01", 34, LIME, True, alpha=ease_out(a4))
        yield f


def seg_problem(fps, sample):
    n = int(4.0 * fps)
    lr = sample["lr"]
    for i in range(n):
        t = i / n
        f = new_frame()
        # slow continuous push-in toward a noisy region
        e = ease_in_out(t)
        size = 620
        im = crop_zoom(lr, lerp(0.5, 0.38, e), lerp(0.5, 0.40, e), lerp(1.0, 0.42, e), size)
        panel(f, im, W // 2, int(H * 0.47), border=BLUE)
        a = ease_out((t - 0.15) / 0.35)
        text(f, (W // 2, int(H * 0.83)), "Noisy.  Low-resolution.  Real inspection data.",
             46, INK, True, alpha=a)
        a2 = ease_out((t - 0.45) / 0.35)
        text(f, (W // 2, int(H * 0.89)),
             "speckle  +  gaussian noise  +  2x downsampling, order unknown",
             30, DIM, alpha=a2)
        text(f, (W // 2, int(H * 0.115)), "THE PROBLEM", 30, LIME, True,
             alpha=ease_out(t / 0.2), spacing_px=6)
        yield f


def seg_sample(fps, s, idx, total):
    """Wipe the degraded input into the restored output, then reveal GT and count metrics."""
    n = int(3.0 * fps)
    size = 540
    lr_up = to_rgb(s["lr"]).resize((size, size), Image.NEAREST)
    pred = to_rgb(s["pred"]).resize((size, size), Image.LANCZOS)
    gt = to_rgb(s["gt"]).resize((size, size), Image.LANCZOS)

    for i in range(n):
        t = i / n
        f = new_frame()
        text(f, (W // 2, int(H * 0.10)), "LIVE RESTORATION", 30, LIME, True,
             alpha=1.0, spacing_px=6)
        text(f, (W - 150, int(H * 0.10)), f"{idx + 1} / {total}", 28, DIM)

        # 0.00-0.55 wipe input -> restored, centred; 0.55+ slide left and reveal GT
        wt = ease_in_out(clamp01(t / 0.55))
        rev = ease_in_out(clamp01((t - 0.58) / 0.30))
        cx_main = int(lerp(W // 2, W * 0.34, rev))
        cy = int(H * 0.47)

        frame_img = wipe(lr_up, pred, wt)
        panel(f, frame_img, cx_main, cy, border=GREEN if wt > 0.99 else BLUE)
        text(f, (cx_main, cy - size // 2 - 34), "degraded  →  restored", 28, DIM,
             alpha=1.0)

        if rev > 0.01:
            # enters from beyond the right edge, so it never overlaps the
            # main panel while that panel is still travelling left
            gx = int(lerp(W * 0.92, W * 0.66, rev))
            gi = gt.resize((int(size * lerp(0.9, 1.0, rev)),) * 2, Image.LANCZOS)
            panel(f, gi, gx, cy, border=LIME)
            text(f, (gx, cy - size // 2 - 34), "ground truth", 28, DIM, alpha=rev)

        # eased count-ups
        ct = ease_out(clamp01((t - 0.62) / 0.33))
        text(f, (W * 0.34, int(H * 0.80)), f"{s['psnr'] * ct:.2f} dB", 62, LIME, True,
             alpha=clamp01(ct * 3))
        text(f, (W * 0.34, int(H * 0.865)), "PSNR", 26, DIM, alpha=clamp01(ct * 3))
        text(f, (W * 0.66, int(H * 0.80)), f"{s['ssim'] * ct:.3f}", 62, LIME, True,
             alpha=clamp01(ct * 3))
        text(f, (W * 0.66, int(H * 0.865)), "SSIM", 26, DIM, alpha=clamp01(ct * 3))

        progress_bar(f, int(H * 0.945), (idx + t) / total)
        yield f


def seg_loss(fps, gravel):
    n = int(5.0 * fps)
    size = 620
    for i in range(n):
        t = i / n
        f = new_frame()
        text(f, (W // 2, int(H * 0.10)), "WHY THE LOSS MATTERS", 30, LIME, True,
             spacing_px=6)

        e = ease_in_out(clamp01(t / 0.62))          # continuous push-in
        frac = lerp(1.0, 0.30, e)
        a = crop_zoom(gravel["l1"], 0.42, 0.55, frac, size)
        b = crop_zoom(gravel["ours"], 0.42, 0.55, frac, size)
        d = ease_in_out(clamp01((t - 0.38) / 0.42))  # cross-dissolve mid push
        panel(f, dissolve(a, b, d), W // 2, int(H * 0.47),
              border=tuple(int(lerp(BLUE[j], GREEN[j], d)) for j in range(3)))

        text(f, (W // 2, int(H * 0.115) + 60), "same image · same model size · only the loss changed",
             30, DIM, alpha=ease_out(t / 0.18))
        lab = "L1 only" if d < 0.5 else "L1 + SSIM + LPIPS"
        col = tuple(int(lerp(BLUE[j], LIME[j], d)) for j in range(3))
        text(f, (W // 2, int(H * 0.80)), lab, 54, col, True)
        lp = lerp(0.397, 0.288, d)
        text(f, (W // 2, int(H * 0.875)), f"LPIPS  {lp:.3f}", 38, DIM)
        text(f, (W // 2, int(H * 0.935)), "texture returns instead of being averaged away",
             28, DIM, alpha=ease_out((t - 0.7) / 0.25))
        yield f


def seg_metrics(fps):
    n = int(5.0 * fps)
    items = [("+{:.2f} dB", 4.05, "PSNR over baseline"),
             ("-{:.0f}%", 59, "LPIPS reduction"),
             ("{:.2f} ms", 4.64, "per image, end to end"),
             ("{:.0f}", 215, "images per second")]
    for i in range(n):
        t = i / n
        f = new_frame()
        text(f, (W // 2, int(H * 0.15)), "RESULTS", 34, LIME, True, spacing_px=8,
             alpha=ease_out(t / 0.12))
        for k, (fmt, val, cap) in enumerate(items):
            start = 0.10 + k * 0.13           # staggered entrance
            p = ease_out(clamp01((t - start) / 0.42))
            if p <= 0:
                continue
            x = int(W * (0.16 + 0.226 * k))
            dy = lerp(28, 0, p)
            text(f, (x, int(H * 0.47 + dy)), fmt.format(val * p), 74, INK, True, alpha=p)
            text(f, (x, int(H * 0.565 + dy)), cap, 27, DIM, alpha=p)
            ImageDraw.Draw(f).rectangle(
                [x - int(70 * p), int(H * 0.615), x + int(70 * p), int(H * 0.615) + 3],
                fill=LIME)
        a = ease_out(clamp01((t - 0.68) / 0.3))
        text(f, (W // 2, int(H * 0.78)),
             "1.16M parameters  ·  4.6 MB  ·  trained in 70 minutes on a laptop GPU",
             32, DIM, alpha=a)
        yield f


def seg_closing(fps):
    n = int(2.5 * fps)
    for i in range(n):
        t = i / n
        f = new_frame()
        a = ease_out(clamp01(t / 0.35))
        text(f, (W // 2, int(H * 0.40)), "FABLERS", 116, INK, True, alpha=a,
             spacing_px=lerp(24, 12, a))
        wr = int(460 * ease_in_out(clamp01((t - 0.2) / 0.35)))
        if wr:
            ImageDraw.Draw(f).rectangle(
                [W // 2 - wr, int(H * 0.505), W // 2 + wr, int(H * 0.505) + 4], fill=LIME)
        a2 = ease_out(clamp01((t - 0.35) / 0.4))
        text(f, (W // 2, int(H * 0.60)), "KLA PS01  ·  AI-Based Restoration of Degraded Images",
             38, DIM, alpha=a2)
        a3 = ease_out(clamp01((t - 0.55) / 0.4))
        text(f, (W // 2, int(H * 0.70)),
             "github.com/anshuman-git01/Fablers-KLA-PS01", 40, LIME, True, alpha=a3)
        yield f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--out", type=Path, default=RESULTS / "demo_video.mp4")
    args = ap.parse_args()

    print("loading models and running inference ...")
    samples, gravel = load_assets(args.samples)
    print(f"  {len(samples)} samples: {', '.join(s['stem'] for s in samples)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def emit(gen, label):
        c = 0
        for fr in gen:
            proc.stdin.write(fr.tobytes())
            c += 1
        print(f"  {label:<18} {c:4d} frames  ({c/args.fps:.1f}s)")
        return c

    total = 0
    print("rendering ...")
    total += emit(seg_opening(args.fps), "opening")
    total += emit(seg_problem(args.fps, samples[0]), "problem")
    for i, s in enumerate(samples):
        total += emit(seg_sample(args.fps, s, i, len(samples)), f"sample {s['stem']}")
    total += emit(seg_loss(args.fps, gravel), "loss comparison")
    total += emit(seg_metrics(args.fps), "headline metrics")
    total += emit(seg_closing(args.fps), "closing")

    proc.stdin.close()
    proc.wait()
    print(f"\n{args.out}  {total} frames  {total/args.fps:.1f}s  "
          f"{args.out.stat().st_size/1e6:.1f} MB")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
