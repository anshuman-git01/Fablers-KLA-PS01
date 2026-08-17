"""Add voice narration and a light music bed to the demo video.

Narration is generated with the macOS `say` engine and placed on a timeline aligned to the
video's segment boundaries. The music bed is synthesised from scratch with numpy, so the
submission carries no third-party audio and no licensing question.

The bed sits about 22 dB under the voice and ducks slightly whenever narration plays, so the
speech stays intelligible without the music disappearing.

Usage:
    python scripts/add_narration.py                    # writes results/demo_video.mp4
    python scripts/add_narration.py --dry-run          # just report clip timings
"""

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import RESULTS  # noqa: E402

SR = 44100
VOICE = "Samantha"
RATE = 178  # words per minute

# (start_seconds, text) — aligned to the video segments:
#   0.0 opening | 3.0 problem | 7.0 restoration (7x3s) | 28.0 loss | 33.0 metrics | 38.0 close
NARRATION = [
    (0.50, "Fablers. A I based image restoration."),
    (3.35, "Images arrive noisy and at half resolution, in an unknown order."),
    (7.30, "Our model restores each image in a single pass."),
    (11.20, "The wipe shows the degraded input becoming the restored output, "
            "with ground truth alongside."),
    (16.40, "Seven validation samples. None were seen during training."),
    (20.70, "Metrics are measured per image."),
    (24.60, "Detail returns, without inventing structure that was never there."),
    (28.40, "Only the loss changed here. L one blurs texture. "
            "Perceptual terms restore it."),
    (33.40, "Four decibels gained, and fifty nine percent lower L PIPS."),
    (38.10, "Fablers. Everything is on GitHub."),
]


def synth_music(dur: float) -> np.ndarray:
    """A quiet ambient pad. Four slow chords, sine partials, gentle detune and drift."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)

    # calm minor progression: Am - F - C - G
    chords = [
        (110.00, 261.63, 329.63),
        (87.31, 220.00, 261.63),
        (130.81, 329.63, 392.00),
        (98.00, 246.94, 293.66),
    ]
    seg = dur / len(chords)
    for i, freqs in enumerate(chords):
        t0 = i * seg
        # long attack and release so chords breathe rather than cut
        env = np.clip((t - t0) / 1.6, 0, 1) * np.clip((t0 + seg + 1.4 - t) / 1.8, 0, 1)
        env = np.clip(env, 0, 1) ** 1.5
        for k, f in enumerate(freqs):
            # slight detune pair per partial gives a slow chorus beat
            drift = 1 + 0.0016 * np.sin(2 * np.pi * (0.05 + 0.011 * k) * t)
            amp = 0.42 / (k + 1.5)
            out += amp * env * (
                np.sin(2 * np.pi * f * drift * t) + np.sin(2 * np.pi * f * 1.002 * t)
            ) / 2

    # faint air layer, low-passed noise
    rng = np.random.default_rng(7)
    air = rng.standard_normal(n)
    kernel = np.ones(220) / 220
    air = np.convolve(air, kernel, mode="same")
    out += 0.05 * air * np.clip(t / 3.0, 0, 1)

    out /= np.max(np.abs(out)) + 1e-9
    # global fade in / out
    fi, fo = int(1.6 * SR), int(2.4 * SR)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= np.linspace(1, 0, fo)
    return out


def say_clip(text: str, path: Path) -> np.ndarray:
    aiff = path.with_suffix(".aiff")
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
         "-ar", str(SR), "-ac", "1", str(path)], check=True)
    with wave.open(str(path), "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    aiff.unlink(missing_ok=True)
    return a.astype(np.float64) / 32768.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=RESULTS / "demo_video.mp4")
    ap.add_argument("--out", type=Path, default=RESULTS / "demo_video.mp4")
    ap.add_argument("--music-gain", type=float, default=0.075)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(args.video)],
        capture_output=True, text=True, check=True).stdout.strip())
    print(f"video duration {dur:.2f}s")

    tmp = Path(subprocess.run(["mktemp", "-d"], capture_output=True, text=True,
                              check=True).stdout.strip())

    voice = np.zeros(int(dur * SR) + SR)
    busy = np.zeros_like(voice)          # 1 where narration is speaking, for ducking
    overruns = []
    for i, (start, txt) in enumerate(NARRATION):
        clip = say_clip(txt, tmp / f"n{i}.wav")
        s = int(start * SR)
        e = min(s + len(clip), len(voice))
        voice[s:e] += clip[: e - s]
        busy[s:e] = 1.0
        end_t = start + len(clip) / SR
        nxt = NARRATION[i + 1][0] if i + 1 < len(NARRATION) else dur
        flag = ""
        if end_t > nxt + 0.05:
            flag = f"  <-- OVERRUNS next cue by {end_t - nxt:.2f}s"
            overruns.append((i, end_t - nxt))
        print(f"  {start:5.2f}s  {len(clip)/SR:4.2f}s  {txt[:52]:<54}{flag}")

    if overruns:
        print(f"\n  {len(overruns)} clip(s) overrun. Shorten the text or raise --rate.")
    if args.dry_run:
        return 1 if overruns else 0

    # smooth the duck envelope so the music dips rather than steps
    k = int(0.35 * SR)
    duck = np.convolve(busy, np.ones(k) / k, mode="same")
    music = synth_music(len(voice) / SR) * args.music_gain
    music *= 1.0 - 0.55 * np.clip(duck, 0, 1)

    mix = voice * 0.92 + music
    peak = np.max(np.abs(mix))
    if peak > 0.97:
        mix *= 0.97 / peak
    print(f"\nmix peak {np.max(np.abs(mix)):.3f}   voice rms {np.sqrt((voice**2).mean()):.4f}"
          f"   music rms {np.sqrt((music**2).mean()):.4f}")

    wav = tmp / "mix.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())

    # never encode in place: render to a temp file, then move over the target
    staged = tmp / "out.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(args.video), "-i", str(wav),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        "-movflags", "+faststart", str(staged),
    ], check=True)
    subprocess.run(["mv", str(staged), str(args.out)], check=True)
    print(f"{args.out}  {args.out.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
