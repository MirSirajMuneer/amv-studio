"""End-to-end + robustness tests for the engine (no GUI needed).

Usage: python tests/test_engine.py <media_dir> <work_dir>
(<media_dir> must contain test_gameplay.mp4 / test_track.wav from make_test_media.py)
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amv_studio.config import EditSettings  # noqa: E402
from amv_studio.errors import AmvError, CancelledError, InputError, MediaError  # noqa: E402
from amv_studio.ffmpeg_utils import find_ffmpeg, probe  # noqa: E402
from amv_studio.pipeline import Pipeline  # noqa: E402
from amv_studio.progress import CancelToken  # noqa: E402

media, work = Path(sys.argv[1]), Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
VIDEO, MUSIC = str(media / "test_gameplay.mp4"), str(media / "test_track.wav")
results = []


def check(name, cond, extra=""):
    results.append(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))


def settings(out, **kw):
    base = dict(durations=[15.0], resolution="720p", quality="Fast", seed=7, output_dir=str(out))
    base.update(kw)
    return EditSettings(**base)


def run(s, videos, music, token=None):
    events = []
    p = Pipeline(s, videos, music, events.append, token)
    try:
        p.run()
        return None, events
    except Exception as e:  # noqa
        return e, events


def leftovers(out):
    return [p.name for p in Path(out).glob(".amv_work_*")]


# ---- 1. bad inputs ------------------------------------------------------------------
corrupt = work / "corrupt.mp4"
corrupt.write_bytes(os.urandom(50_000))
e, ev = run(settings(work / "o1"), [str(corrupt)], MUSIC)
check("corrupt video -> MediaError with friendly text", isinstance(e, MediaError) and "readable" in e.message, str(e)[:70])
check("error event emitted", any(x["kind"] == "error" for x in ev))

e, _ = run(settings(work / "o2"), [VIDEO], str(work / "missing.wav"))
check("missing music -> MediaError", isinstance(e, MediaError), str(e)[:60])

e, _ = run(settings(work / "o3"), [VIDEO], VIDEO)   # video-only file as 'music'
check("music without audio track -> MediaError", isinstance(e, MediaError) and "no audio" in e.message, str(e)[:60])

e, _ = run(settings(work / "o4"), [], MUSIC)
check("no videos -> InputError", isinstance(e, InputError))

e, _ = run(settings("/proc/amv_no_write"), [VIDEO], MUSIC)
check("unwritable output folder -> InputError", isinstance(e, InputError), str(e)[:60])

# ---- 2. cancel mid-run, clean up ---------------------------------------------------------
out = work / "o5"
tok = CancelToken()
s = settings(out, durations=[50.0], quality="High")


def canceller(events):
    for _ in range(600):
        if any(x["kind"] == "step_progress" and x["step"] == "render" and x["frac"] > 0.1 for x in events):
            tok.cancel()
            return
        time.sleep(0.05)


events = []
p = Pipeline(s, [VIDEO], MUSIC, events.append, tok)
threading.Thread(target=canceller, args=(events,), daemon=True).start()
t0 = time.time()
err = None
try:
    p.run()
except Exception as ex:  # noqa
    err = ex
check("cancel -> CancelledError", isinstance(err, CancelledError), type(err).__name__)
check("cancel returned quickly", time.time() - t0 < 60, f"{time.time() - t0:.1f}s")
check("cancel left no temp folders", not leftovers(out), str(leftovers(out)))
check("cancel left no output files", not list(out.glob("*.mp4")))
time.sleep(0.5)
stray = subprocess.run(["pgrep", "-fc", "ffmpeg.*amv_work"], capture_output=True, text=True).stdout.strip()
check("no stray ffmpeg processes", stray in ("0", ""), stray)

# ---- 3. music shorter than requested length -> shortened, warns ---------------------------------
out = work / "o6"
e, ev = run(settings(out, durations=[120.0], resolution="720p"), [VIDEO], MUSIC)
files = sorted(out.glob("*.mp4"))
dur = probe(str(files[0])).duration if files else 0
check("long request on 70 s song -> shortened to song length", e is None and abs(dur - 70) < 0.6, f"{dur:.2f}s")
check("warning logged about shortened output",
      any(x["kind"] == "log" and "shortened" in x["message"] for x in ev))

# ---- 4. footage shorter than the edit needs (moments reused) -----------------------------------------
short = work / "short10.mp4"
subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-ss", "0", "-t", "10", "-i", VIDEO, "-c", "copy", str(short)], check=True)
out = work / "o7"
e, ev = run(settings(out, durations=[25.0]), [str(short)], MUSIC)
files = sorted(out.glob("*.mp4"))
check("10 s footage -> 25 s AMV", e is None and files and abs(probe(str(files[0])).duration - 25) < 0.1, str(e)[:80])

# ---- 5. other formats / options ------------------------------------------------------------------------
out = work / "o8"
e, _ = run(settings(out, aspect="16:9", fps=60, framing="blur_fit", quality="High", look="Cinematic",
                    durations=[15.0], variations=2), [VIDEO], MUSIC)
files = sorted(out.glob("*.mp4"))
ok = e is None and len(files) == 2
if ok:
    i = probe(str(files[0]))
    ok = (i.width, i.height) == (1280, 720) and abs(i.fps - 60) < 0.5 and abs(i.duration - 15) < 0.1 and i.has_audio
check("16:9 / 60fps / High / 2 variations", ok, str(e)[:80] if e else "")

out = work / "o9"
e, _ = run(settings(out, aspect="1:1", resolution="1080p", durations=[15.0], transitions=[], slowmo=False,
                    zoom_amount=0.0), [VIDEO, str(short)], MUSIC)
files = sorted(out.glob("*.mp4"))
ok = e is None and len(files) == 1 and (probe(str(files[0])).width, probe(str(files[0])).height) == (1080, 1080)
check("1:1 1080p, no effects, 2 source videos", ok, str(e)[:80] if e else "")

# ---- 6. v1.1 features ------------------------------------------------------------------------------------------
GAME = str(media / "test_gameplay_audio.mp4") if (media / "test_gameplay_audio.mp4").exists() else VIDEO
out = work / "o10"
e, ev = run(settings(out, durations=[12.0], custom_size="640x1138", fx=["grain", "vhs", "glow", "chroma", "trails"],
                     transitions=["glitch", "rgb", "dip", "strobe", "tilt", "roll", "invert"],
                     title_text="TEST TITLE", outro_text="THE END", watermark_text="@me", game_audio=0.3,
                     sync_nudge_ms=40, base_zoom=1.2), [GAME], MUSIC)
files = sorted(out.glob("*.mp4"))
ok = e is None and len(files) == 1
if ok:
    i = probe(str(files[0]))
    ok = (i.width, i.height) == (640, 1138) and abs(i.duration - 12) < 0.15 and i.has_audio
check("custom size + all-new fx/transitions + text + game audio + nudge", ok, str(e)[:90] if e else "")
check("no warnings in that run", not [x for x in ev if x["kind"] == "log" and x.get("level") == "WARNING"],
      "; ".join(x["message"][:60] for x in ev if x["kind"] == "log" and x.get("level") == "WARNING")[:150])
check("edit report written", bool(list((out / "edit_reports").glob("*.txt"))))

out = work / "o11"
e, _ = run(settings(out, durations=[10.0]), [VIDEO], [MUSIC, MUSIC])
check("multi-song job -> one file per song", e is None and len(list(out.glob("*.mp4"))) == 2, str(e)[:80] if e else "")

out = work / "o12"
e, _ = run(settings(out, durations=[10.0], resolution="1080p"), [VIDEO], MUSIC)
# preview = fast low-res draft in <out>/previews
pe = Pipeline(settings(out / "pv", durations=[10.0], resolution="1080p"), [VIDEO], MUSIC, None, None, preview=True)
try:
    pe.run(); pf = sorted((out / "pv" / "previews").glob("*.mp4"))
    pi = probe(str(pf[0])) if pf else None
    check("preview draft is 480p", bool(pf) and min(pi.width, pi.height) == 480,
          f"{pi.width}x{pi.height}" if pf else "none")
except Exception as ex:  # noqa
    err = ex
check("cancel -> CancelledError", isinstance(err, CancelledError), type(err).__name__)
check("cancel returned quickly", time.time() - t0 < 60, f"{time.time() - t0:.1f}s")
check("cancel left no temp folders", not leftovers(out), str(leftovers(out)))
check("cancel left no output files", not list(out.glob("*.mp4")))
time.sleep(0.5)
stray = subprocess.run(["pgrep", "-fc", "ffmpeg.*amv_work"], capture_output=True, text=True).stdout.strip()
check("no stray ffmpeg processes", stray in ("0", ""), stray)

# ---- 3. music shorter than requested length -> shortened, warns ---------------------------------
out = work / "o6"
e, ev = run(settings(out, durations=[120.0], resolution="720p"), [VIDEO], MUSIC)
files = sorted(out.glob("*.mp4"))
dur = probe(str(files[0])).duration if files else 0
check("long request on 70 s song -> shortened to song length", e is None and abs(dur - 70) < 0.6, f"{dur:.2f}s")
check("warning logged about shortened output",
      any(x["kind"] == "log" and "shortened" in x["message"] for x in ev))

# ---- 4. footage shorter than the edit needs (moments reused) -----------------------------------------
short = work / "short10.mp4"
subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-ss", "0", "-t", "10", "-i", VIDEO, "-c", "copy", str(short)], check=True)
out = work / "o7"
e, ev = run(settings(out, durations=[25.0]), [str(short)], MUSIC)
files = sorted(out.glob("*.mp4"))
check("10 s footage -> 25 s AMV", e is None and files and abs(probe(str(files[0])).duration - 25) < 0.1, str(e)[:80])

# ---- 5. other formats / options ------------------------------------------------------------------------
out = work / "o8"
e, _ = run(settings(out, aspect="16:9", fps=60, framing="blur_fit", quality="High", look="Cinematic",
                    durations=[15.0], variations=2), [VIDEO], MUSIC)
files = sorted(out.glob("*.mp4"))
ok = e is None and len(files) == 2
if ok:
    i = probe(str(files[0]))
    ok = (i.width, i.height) == (1280, 720) and abs(i.fps - 60) < 0.5 and abs(i.duration - 15) < 0.1 and i.has_audio
check("16:9 / 60fps / High / 2 variations", ok, str(e)[:80] if e else "")

out = work / "o9"
e, _ = run(settings(out, aspect="1:1", resolution="1080p", durations=[15.0], transitions=[], slowmo=False,
                    zoom_amount=0.0), [VIDEO, str(short)], MUSIC)
files = sorted(out.glob("*.mp4"))
ok = e is None and len(files) == 1 and (probe(str(files[0])).width, probe(str(files[0])).height) == (1080, 1080)
check("1:1 1080p, no effects, 2 source videos", ok, str(e)[:80] if e else "")

# ---- 6. v1.1 features ------------------------------------------------------------------------------------------
GAME = str(media / "test_gameplay_audio.mp4") if (media / "test_gameplay_audio.mp4").exists() else VIDEO
out = work / "o10"
e, ev = run(settings(out, durations=[12.0], custom_size="640x1138", fx=["grain", "vhs", "glow", "chroma", "trails"],
                     transitions=["glitch", "rgb", "dip", "strobe", "tilt", "roll", "invert"],
                     title_text="TEST TITLE", outro_text="THE END", watermark_text="@me", game_audio=0.3,
                     sync_nudge_ms=40, base_zoom=1.2), [GAME], MUSIC)
files = sorted(out.glob("*.mp4"))
ok = e is None and len(files) == 1
if ok:
    i = probe(str(files[0]))
    ok = (i.width, i.height) == (640, 1138) and abs(i.duration - 12) < 0.15 and i.has_audio
check("custom size + all-new fx/transitions + text + game audio + nudge", ok, str(e)[:90] if e else "")
check("no warnings in that run", not [x for x in ev if x["kind"] == "log" and x.get("level") == "WARNING"],
      "; ".join(x["message"][:60] for x in ev if x["kind"] == "log" and x.get("level") == "WARNING")[:150])
check("edit report written", bool(list((out / "edit_reports").glob("*.txt"))))

out = work / "o11"
e, _ = run(settings(out, durations=[10.0]), [VIDEO], [MUSIC, MUSIC])
check("multi-song job -> one file per song", e is None and len(list(out.glob("*.mp4"))) == 2, str(e)[:80] if e else "")

out = work / "o12"
e, _ = run(settings(out, durations=[10.0], resolution="1080p"), [VIDEO], MUSIC)
# preview = fast low-res draft in <out>/previews
pe = Pipeline(settings(out / "pv", durations=[10.0], resolution="1080p"), [VIDEO], MUSIC, None, None, preview=True)
try:
    pe.run(); pf = sorted((out / "pv" / "previews").glob("*.mp4"))
    check("preview draft is 480p", bool(pf) and probe(str(pf[0])).width == 270 or probe(str(pf[0])).height == 480,
          f"{probe(str(pf[0])).width}x{probe(str(pf[0])).height}" if pf else "none")
except Exception as ex:  # noqa
    check("preview draft is 480p", False, str(ex)[:80])

out = work / "o13"
e, _ = run(settings(out, durations=[8.0], slowmo_smoothing="Motion", quality="Fast", seed=3), [VIDEO], MUSIC)
check("Motion smoothing mode renders", e is None and len(list(out.glob("*.mp4"))) == 1, str(e)[:80] if e else "")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
