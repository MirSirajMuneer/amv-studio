#!/usr/bin/env python3
"""AMV Studio entry point.

    python main.py                      -> opens the GUI
    python main.py --cli --video a.mp4 --music song.mp3 --duration 30 60    -> headless
    python main.py --check              -> verify the installation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from amv_studio import config  # noqa: E402
from amv_studio.config import (ASPECTS, FPS_CHOICES, FRAMINGS, FX, LOOKS, PLATFORMS, PRESETS, QUALITIES,  # noqa: E402
                               RESOLUTIONS, TRANSITIONS, EditSettings)


def check_install() -> int:
    ok = True
    print(f"{config.APP_NAME} {config.VERSION} - installation check\n")
    print(f"Python      : {sys.version.split()[0]}")
    for mod in ("numpy", "scipy", "cv2", "PIL", "imageio_ffmpeg", "customtkinter"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}: OK ({getattr(m, '__version__', 'n/a')})")
        except Exception as e:
            ok = False
            print(f"{mod:<12}: MISSING ({e.__class__.__name__}) -> pip install -r requirements.txt")
    try:
        from amv_studio.ffmpeg_utils import detect_encoders, find_ffmpeg
        print(f"ffmpeg      : {find_ffmpeg()}")
        hw = detect_encoders()
        print(f"encoders    : {', '.join(hw) if hw else 'CPU (libx264) only'}")
    except Exception as e:
        ok = False
        print(f"ffmpeg      : NOT FOUND ({e})")
    try:
        import tkinter  # noqa: F401
        print("tkinter     : OK")
    except Exception:
        ok = False
        print("tkinter     : MISSING (Linux: sudo apt install python3-tk)")
    try:
        import tkinterdnd2  # noqa: F401
        print("tkinterdnd2 : OK (drag & drop enabled)")
    except Exception:
        print("tkinterdnd2 : not installed (optional - enables drag & drop of files)")
    print("\nAll good!" if ok else "\nSome things need attention (see above).")
    return 0 if ok else 1


def run_cli(args: argparse.Namespace) -> int:
    from amv_studio.errors import AmvError, CancelledError
    from amv_studio.pipeline import STEPS, Pipeline

    if not args.video or not args.music:
        print("CLI mode needs --video and --music.")
        return 2
    s = EditSettings()
    s.apply_preset(args.preset)
    if args.platform != "None":
        s.apply_platform(args.platform)
    s.durations = args.duration or [30.0]
    s.variations, s.seed, s.output_dir = args.variations, args.seed, args.out or ""
    s.use_cache = not args.no_cache
    if args.aspect:
        s.aspect = args.aspect
    if args.res:
        s.resolution = args.res
    if args.size:
        s.custom_size = args.size
    if args.fps:
        s.fps = args.fps
    if args.look:
        s.look = args.look
    if args.quality:
        s.quality = args.quality
    if args.cpu:
        s.encoder = "CPU"
    if args.framing:
        s.framing = args.framing
    if args.story:
        s.order_mode = "story"
    if args.fx is not None:
        s.fx = args.fx
    if args.transitions is not None:
        s.transitions = args.transitions
    if args.motion:
        s.slowmo_smoothing = "Motion"
    if args.title:
        s.title_text = args.title
    if args.outro:
        s.outro_text = args.outro
    if args.watermark:
        s.watermark_text = args.watermark
    if args.game_audio is not None:
        s.game_audio = args.game_audio
    s.clamp()
    labels = {sid: lab for sid, lab, _ in STEPS}
    last = {"step": None, "pct": -1}

    def cb(ev: dict) -> None:
        k = ev["kind"]
        if k == "step_progress":
            pct = int(ev["frac"] * 100)
            if ev["step"] != last["step"] or pct != last["pct"]:
                last["step"], last["pct"] = ev["step"], pct
                print(f"\r[{labels[ev['step']]:<24}] {pct:3d}%  {ev.get('message', '')[:50]:<50}", end="", flush=True)
        elif k == "step_done":
            print(f"\r[{labels[ev['step']]:<24}] done  {ev.get('message', '')[:50]:<50}")
        elif k == "log" and ev.get("level") in ("WARNING", "ERROR"):
            print(f"\n  ! {ev['message']}")
        elif k == "output":
            print(f"  saved: {ev['path']}")

    try:
        Pipeline(s, args.video, args.music, cb, preview=args.preview).run()
    except CancelledError:
        print("\nCancelled.")
        return 130
    except AmvError as e:
        print(f"\nERROR: {e.user_text()}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}  (see log in ~/.amv_studio/logs)")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=f"{config.APP_NAME} {config.VERSION} - automatic beat-synced AMV maker")
    p.add_argument("--cli", action="store_true", help="run headless (no GUI)")
    p.add_argument("--check", action="store_true", help="verify installation and exit")
    p.add_argument("--video", nargs="+", help="input video file(s)")
    p.add_argument("--music", nargs="+", help="music file(s) - one output set per song")
    p.add_argument("--duration", nargs="+", type=float, help="output length(s) in seconds, e.g. 25 60 120")
    p.add_argument("--out", help="output folder")
    p.add_argument("--preset", default="GTA Action (Recommended)", choices=list(PRESETS))
    p.add_argument("--platform", default="None", choices=["None"] + [k for k in PLATFORMS if k != "None"])
    p.add_argument("--aspect", choices=ASPECTS)
    p.add_argument("--res", choices=RESOLUTIONS)
    p.add_argument("--size", help="custom output size WxH, e.g. 1440x2560")
    p.add_argument("--fps", type=int, choices=FPS_CHOICES)
    p.add_argument("--look", choices=LOOKS)
    p.add_argument("--quality", choices=QUALITIES)
    p.add_argument("--framing", choices=FRAMINGS)
    p.add_argument("--fx", nargs="*", choices=FX, help="effects list (empty = none)")
    p.add_argument("--transitions", nargs="*", choices=TRANSITIONS)
    p.add_argument("--variations", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", help="title text overlay")
    p.add_argument("--outro", help="outro text")
    p.add_argument("--watermark", help="watermark text")
    p.add_argument("--game-audio", type=float, help="0-1 level of original game sound under the music")
    p.add_argument("--motion", action="store_true", help="motion-interpolated slow-mo (very slow, smoothest)")
    p.add_argument("--preview", action="store_true", help="fast low-res draft")
    p.add_argument("--story", action="store_true", help="keep footage in chronological order")
    p.add_argument("--cpu", action="store_true", help="force CPU encoding")
    p.add_argument("--no-cache", action="store_true", help="ignore cached video analysis")
    args = p.parse_args()

    if args.check:
        return check_install()
    if args.cli:
        return run_cli(args)
    try:
        from amv_studio.gui.app import run_gui
    except ImportError as e:
        print(f"Could not start the GUI: {e}\n\nInstall the requirements first:\n    pip install -r requirements.txt\n"
              "On Linux you may also need:  sudo apt install python3-tk\n"
              "You can still use the command line:  python main.py --cli --help")
        return 1
    run_gui()
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
