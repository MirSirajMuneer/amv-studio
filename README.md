# AMV Studio 1.1

Give it long gameplay footage (GTA 6 or anything else) and one or more songs. It finds the exciting moments, cuts them **on the beat**, adds smooth slow-mo speed ramps, zoom punches, 12 transitions, 9 stackable effects, 10 colour looks and text overlays, then renders finished videos of **any length from 5 s to 10 min** (25 s, 60 s, 1:30, 3:00 … or custom) in 9:16, 16:9 or 1:1 at **480p – 4K (2160p)**, 24/30/60 fps, or a **custom WxH size**.

See `BLUEPRINT.md` for the design.

## Install (once)

1. Install **Python 3.10 or newer** (Windows: tick "Add Python to PATH").
2. Double-click **`run_windows.bat`** (Windows) or run **`./run_mac_linux.sh`** — it creates a virtual environment, installs `requirements.txt`, and starts the app.

Manual way:

```bash
pip install -r requirements.txt
python main.py
```

ffmpeg is bundled through `imageio-ffmpeg`. Linux, if the window won't open: `sudo apt install python3-tk`.
Check your setup: `python main.py --check`.

## What's new in 1.1

* **Smoother interface** – the video engine now runs in its own process (the window never freezes), progress is batched, controls are organised in tabs (Media · Style · Effects · Output · Text & Audio), UI size menu (80–130 %).
* **Output scale** – 480p / 720p / 1080p / 1440p / 4K, 24/30/60 fps, custom `WxH`, platform presets (TikTok, Shorts, Reels, YouTube, YouTube 60 fps), footage zoom, three framing modes (fill / blurred fit / black bars).
* **New effects** – glow/bloom, chromatic aberration, VHS retro, beat flicker, extra sharpen, ghost trails, cinema bars, film grain, vignette. Strength slider.
* **New transitions** – glitch, whip-tilt, roll, strobe, invert flash, RGB split, dip-to-black (plus flash, zoom punch, shake, blur zoom, whip pan). Tick any combination.
* **New looks** – Noir, Warm Sunset, Cold Steel, Teal & Orange, Clean (plus Natural, Vivid, Cinematic, Neon, Gritty).
* **More control** – preview (fast 480p draft, final render re-uses the same edit), sync nudge (± ms), half/double-time beat grid, beat-pulse mode, min/max shot length, diversity, shuffle/story/energy ordering, game-audio under the music, title / outro / watermark text, multiple songs in one job, project files (`.amvproj`), your own saved presets, edit report, support bundle.
* **Always-smooth slow-mo** – slowed shots are frame-blended even in *Fast* quality (optional *Motion interpolation* is much slower but smoothest).
* **Faster** – clips render in parallel; analysis is cached.

## Use

1. **Media** – add videos (or drag & drop if `tkinterdnd2` is installed) and one or more songs.
2. **Style** – pick a preset, tune the sliders.
3. **Effects** – tick transitions and effects.
4. **Output** – lengths, format, resolution, fps, folder.
5. **Preview** for a quick draft, then **⚡ GENERATE**. **Cancel** stops cleanly.

Headless / batch:

```bash
python main.py --cli --video a.mp4 b.mp4 --music song.mp3 --duration 25 60 --out ./shorts \
       --platform "YouTube Shorts" --fx grain glow --transitions glitch rgb whip --title "GTA 6"
python main.py --cli --help
```

## Tips

* Quality **Balanced** is the sweet spot; **Fast** is for drafts. Use *Blend* smoothing for slow-mo; *Motion* only for a final hero edit.
* Songs with a clear beat and a drop work best. Longer footage = better picks.
* Slow computer? 720p + Fast. GPU? Leave encoder on Auto.

## Troubleshooting

* Log: `~/.amv_studio/logs/amv_studio.log` (button **Log file**, or **Support** to bundle it).
* "ffmpeg not found" → `pip install imageio-ffmpeg`.
* Make sure you have the rights to the footage and music you use.

## Tests

```bash
python tests/make_test_media.py media
python tests/test_effects_matrix.py media work
python tests/test_engine.py media work
python tests/test_gui_logic.py media out
```

## Building a Windows .exe

`build_exe.bat` (PyInstaller). Untested by the author – verify on a clean PC.

## License & disclaimer

Released under the **MIT License** (free to use, modify and share) – see `LICENSE`.
AMV Studio is an independent project, not affiliated with or endorsed by Rockstar Games, Take-Two Interactive or any game publisher. You are responsible for having the rights to the footage and music you edit. FFmpeg (bundled via `imageio-ffmpeg`) has its own license.
