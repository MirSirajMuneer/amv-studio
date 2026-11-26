"""Generate synthetic test media: a 60 s video (action / static / black / flashes) and a 120 BPM track.

Usage: python tests/make_test_media.py <output_dir>
Ground truth: tempo 120 BPM, beats at k*0.5 s, drop at 16.0 s, black frames at 32-38 s, bars at 12-20 s.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amv_studio.ffmpeg_utils import find_ffmpeg  # noqa: E402


def make_video(path: Path) -> None:
    ff = find_ffmpeg()
    W, H, R = 1280, 720, 30
    srcs = [
        ("testsrc2=s=%dx%d:r=%d:d=12" % (W, H, R), ""),
        ("smptebars=s=%dx%d:r=%d:d=8" % (W, H, R), ""),
        ("mandelbrot=s=%dx%d:r=%d" % (W, H, R), ",trim=duration=12"),
        ("color=c=black:s=%dx%d:r=%d:d=6" % (W, H, R), ""),
        ("testsrc2=s=%dx%d:r=%d:d=12" % (W, H, R),
         ",hue=h='t*60',eq=brightness='0.7*lt(mod(t,1.5),0.08)':eval=frame"),
        ("mandelbrot=s=%dx%d:r=%d" % (W, H, R), ",trim=duration=10"),
    ]
    cmd = [ff, "-y", "-loglevel", "error"]
    for s, _ in srcs:
        cmd += ["-f", "lavfi", "-i", s]
    fc = []
    for i, (_, extra) in enumerate(srcs):
        fc.append(f"[{i}:v]format=yuv420p,setpts=PTS-STARTPTS{extra},fps={R},setsar=1[v{i}]")
    fc.append("".join(f"[v{i}]" for i in range(len(srcs))) + f"concat=n={len(srcs)}:v=1:a=0[v]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


def make_music(path: Path, dur: float = 70.0, sr: int = 44100, bpm: float = 120.0) -> None:
    n = int(dur * sr)
    y = np.zeros(n, np.float32)
    rng = np.random.default_rng(1)
    beat = 60.0 / bpm

    def add(t: float, sig: np.ndarray, gain: float) -> None:
        i = int(t * sr)
        j = min(n, i + len(sig))
        if i < n:
            y[i:j] += gain * sig[: j - i]

    tk = np.arange(int(0.28 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 * tk + 70 * (1 - np.exp(-tk * 18)) / 18 * 0 + 60 * np.exp(-tk * 25) / 25)) \
        * np.exp(-tk * 14)
    th = np.arange(int(0.05 * sr)) / sr
    hat = rng.standard_normal(len(th)).astype(np.float32) * np.exp(-th * 90)
    ts = np.arange(int(0.14 * sr)) / sr
    snare = rng.standard_normal(len(ts)).astype(np.float32) * np.exp(-ts * 28)
    drop_t = 16.0
    k = 0
    t = 0.0
    while t < dur - 0.3:
        loud = t >= drop_t
        mid = 8.0 <= t < drop_t
        add(t + beat / 2, hat, 0.25 if (loud or mid) else 0.12)
        if mid or loud:
            add(t, kick, 0.8 if loud else 0.45)
        if loud and k % 2 == 1:
            add(t, snare, 0.5)
        k += 1
        t += beat
    # sustained bass + pad in the drop section
    tt = np.arange(n) / sr
    env = np.clip((tt - drop_t) * 4, 0, 1)
    y += (0.18 * np.sin(2 * np.pi * 55 * tt) * env).astype(np.float32)
    y += (0.06 * np.sin(2 * np.pi * 220 * tt) * env * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * tt))).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    wavfile.write(str(path), sr, (y * 32767).astype(np.int16))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "test_media")
    out.mkdir(parents=True, exist_ok=True)
    make_video(out / "test_gameplay.mp4")
    make_music(out / "test_track.wav")
    # same footage with a noisy "game sound" track (engine hum + bursts) for the audio-mix tests
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(out / "test_gameplay.mp4"), "-f", "lavfi",
                    "-i", "anoisesrc=d=60:c=pink:a=0.25,lowpass=f=900,tremolo=f=3:d=0.7", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out / "test_gameplay_audio.mp4")], check=True)
    print("created", out)
