# AMV Studio — Software Blueprint (v1.1)

Automated, beat-synced AMV / short-video maker. You give it long gameplay footage
(e.g. GTA 6) and a music track; it finds the best moments, cuts them on the beat,
adds slow-mo speed ramps, zoom punches, transitions and a colour look, and renders
finished 9:16 / 16:9 / 1:1 shorts of any length (15 s – 3 min).

---

## 1. Goals and non-goals

**Goals**
1. One-click automation: *video(s) + music → finished short*.
2. Edits that actually follow the music: cuts land on beats, bigger moments land on the loud / drop sections.
3. Pick "cool parts" automatically (action, explosions, speed, vivid scenes) and skip menus, loading screens, black frames.
4. Friendly GUI: every pipeline step has its own progress bar, a live log, and a Cancel button.
5. Robust: clear error messages, automatic fallbacks, no stuck temp files, full log file for support.
6. Easy install: `pip install -r requirements.txt` (ffmpeg is bundled through `imageio-ffmpeg`).

**Non-goals (v1.1)**: manual timeline editing / per-clip swap & lock, AI object/face recognition, lyrics/captions, licensing & payments.

---

## 2. User workflow

```
 Add video(s) ─┐
 Add music ────┼─► Pick lengths (25/50/60/90/120/180 s …) ─► Pick style preset ─► GENERATE
 Pick folder ──┘                                                     │
                                                                     ▼
        Step bars: Check files → Music beats → Find moments → Design edit → Render → Mix & finish
                                                                     │
                                                                     ▼
                                        MP4 files in output folder  (+ "Open folder" button)
```

---

## 3. System architecture

```
main.py ──► GUI (customtkinter)  ─┐                  ┌── ffmpeg_utils  (locate/probe/run/encoders)
        └─► CLI (argparse)  ──────┴─► Pipeline ──────┼── audio_analysis (beats, energy, drops)
                                       │             ├── video_analysis (motion/fire/flash/scenes → moments)
        events (queue) ◄───────────────┤             ├── planner        (beat grid → slots → clip plans)
        CancelToken  ──────────────────┘             ├── effects        (plan → ffmpeg filter graph)
                                                     └── renderer       (render clips, concat, mux music)
```

Layers never import upward: `gui → pipeline → {analysis, planner, renderer} → {effects, ffmpeg_utils} → {config, errors}`.
The GUI runs the pipeline in a worker thread and receives progress events via a `queue.Queue`.

### 3.1 Module map

| Module | Responsibility |
|---|---|
| `config.py` | `EditSettings` dataclass, style presets, looks, resolutions, JSON persistence |
| `errors.py` | `AmvError` hierarchy with user-friendly message + hint |
| `logger.py` | Rotating file log in `~/.amv_studio/logs` + in-memory hook for the GUI |
| `ffmpeg_utils.py` | Find ffmpeg/ffprobe, probe media, run with progress+cancel, raw frame/audio streaming, GPU encoder detection |
| `audio_analysis.py` | Decode → spectral flux → tempo → DP beat tracker → intensity curve → drops → best song window |
| `video_analysis.py` | Low-res frame stream → optical flow, fire/explosion, brightness flashes, sharpness, scene cuts → score → moments; disk cache |
| `planner.py` | Beat grid → slots (1/2/4 beats) → assign moments → speed ramps, zooms, transitions → `EditPlan` |
| `effects.py` | `ClipPlan` → ffmpeg `filter_complex` (ramp, framing, zoom/pulse, shake, glitch, whip, flash, look, fades) |
| `renderer.py` | Render every clip (with retry/fallback), concat, mux + loudness-normalise music |
| `pipeline.py` | Orchestrates steps, aggregates progress, cache, temp dir, cleanup, multi-output |
| `gui/app.py` | Window, settings panels, step progress rows, log, run/cancel |

---

## 4. Processing pipeline

| # | Step | What happens | Progress source |
|---|---|---|---|
| 1 | **Check files** | exist/readable, probe with ffprobe, has video / has audio, disk space, ffmpeg present, output folder writable | per file |
| 2 | **Music beats** | decode to mono 22.05 kHz → onset envelope → tempo → beat tracking → intensity curve → drop detection | decode bytes + loops |
| 3 | **Find moments** | stream 6 fps / 192 px frames from ffmpeg → features per sample → score → scene cuts → peak moments. Results cached on disk, so a 2nd run (other length / music) is instant | frames processed / estimate |
| 4 | **Design edit** | choose song window, beat grid, slots, assign moments, ramps, zooms, transitions | per output |
| 5 | **Render clips** | one ffmpeg process per slot with its own filter graph | clips done / total (+ ffmpeg `out_time`) |
| 6 | **Mix & finish** | concat (stream copy), attach music with fades + loudnorm, `+faststart` | ffmpeg `out_time` |

Several lengths/variations (e.g. 25 s + 60 s + 120 s) share steps 1–3 and loop over 4–6.

---

## 5. Core algorithms

### 5.1 Music analysis (pure NumPy/SciPy — no librosa/numba needed)
1. Decode audio with ffmpeg → mono float32 @ 22 050 Hz.
2. STFT (n_fft 1024, hop 256 → ~86 frames/s), processed in chunks (low RAM).
3. **Onset envelope** = positive log-magnitude spectral flux, locally normalised.
4. **Tempo** = autocorrelation of the envelope in 60–200 BPM with a log-normal prior around 120 BPM.
5. **Beats** = Ellis (2007) dynamic-programming beat tracker, then snapped to the sharpest nearby onset (±60 ms).
6. **Intensity curve** (0..1) = smoothed loudness (60 %) + onset density (40 %); sampled at every beat.
7. **Drops** = positions where the next 2 s of low-band + loudness energy exceeds the previous 4 s by a margin; snapped to a beat.
8. **Best window**: for a requested length D, choose the beat-aligned window with the highest mean intensity (+ bonus if it contains a drop early). Variation *k* takes the *k*-th best non-overlapping window.

### 5.2 Video analysis ("find the cool parts")
Per sample (6 fps, 192 px wide):

| Feature | Meaning |
|---|---|
| `motion` | mean Farneback optical-flow magnitude (camera / action speed) |
| `peak` | 97th-percentile flow (localised action: a car crash, a jump) |
| `fire` | fraction of bright orange/yellow pixels (explosions, muzzle flash, fire) |
| `flash` | positive brightness jump (explosion flashes, lightning) |
| `sat`, `sharp`, `contrast` | vivid, in-focus, non-flat frames |
| `cx` | horizontal centroid of motion (used for smart-crop) |
| penalties | very dark / flat frames (menus, loading, black) and near-static frames |

`score = 0.34·motion + 0.18·peak + 0.14·fire + 0.12·flash + 0.10·sat + 0.07·sharp + 0.05·contrast`
(each robust-normalised to p5–p95 per file), multiplied by penalties and smoothed over ~1.2 s.

**Scene cuts**: histogram-correlation drop + frame difference above an adaptive threshold.
**Moments** = local maxima of the smoothed score (≥ 2 s apart, above the 40th percentile) with the surrounding scene bounds, so a clip never contains an accidental cut.
Fallback: if too few moments are found, evenly spaced samples are used.

### 5.3 Edit planning
1. **Beat grid** inside the chosen song window.
2. **Slot length** (in beats: 1, 2 or 4) from the *Cut speed* setting, tempo, and local intensity — calm parts get long shots, loud parts get fast cuts. A detected drop becomes a 2-beat "hero" slot.
3. **Assignment**
   - *Energy match* (default): best moments → highest-intensity slots.
   - *Story order*: top moments kept in chronological order.
   Variations add ±15 % score noise (seeded) so every output differs.
4. **Source window** is placed so the action peak lands ~35 % into the shot, clamped inside the scene.
5. **Speed ramp** (slow-mo): `fast → slow (0.3–0.6×) → fast`, 30 / 40 / 30 % of the slot, ending exactly on the next beat. Forced on drop slots; never on two slots in a row.
6. **Zoom**: beat-pulse (pop on every beat), slow push / pull, punch-in on cuts.
7. **Transition-in** per slot (not first): `flash`, `punch`, `shake`, `glitch`, `whip`, chosen by effect intensity × slot intensity, never repeating twice in a row.
8. First clip fades in from black, last fades out; music fades in/out.
9. All slot boundaries are snapped to **output frames**, so the final video length equals the target to the frame.

### 5.4 Rendering
* Each slot → one ffmpeg run, one `filter_complex`:
  `trim/setpts (ramp) → concat → tpad → scale+crop (fill) or blur-fit → zoom/pulse/shake → colour look → glitch/whip/flash → fades → yuv420p`.
* Slow-mo smoothing (`minterpolate`) is only applied if the source has too few frames (quality-dependent).
* Encoder: libx264 by default; **Auto** tests NVENC / AMF / QSV / VideoToolbox with a 0.2 s test encode and uses the first that works.
* Assembly: concat demuxer with `-c:v copy`, music via `-ss/-t`, `afade`, `loudnorm`, AAC 256 k.

---

## 6. Effect catalogue

| Effect | Implementation |
|---|---|
| Slow-mo / speed ramp | `trim` + `setpts` per segment, `concat`, optional `minterpolate` |
| Beat-pulse zoom | `scale` with `eval=frame` and `pow(1-mod(t,P)/P,3)` |
| Push / pull zoom | `scale eval=frame` ease over the clip |
| Punch-in | decaying zoom term in the first 0.35 s |
| Shake | `crop` with decaying sin/cos offsets |
| Glitch | `rgbashift` + noise gated by `enable='lt(t,…)'` |
| Whip / blur | `gblur` for the first ~0.2 s + punch |
| Flash | `fade` from white |
| Looks | Natural, Vivid, Cinematic, Neon, Gritty (`eq`, `colorbalance`, `vignette`, `unsharp`) |
| Framing | Fill (smart crop to motion) or Blur-fit (blurred background) |

---

## 7. Data models

```text
MediaInfo      path, duration, width, height, fps, has_video, has_audio
MusicAnalysis  duration, tempo, beats[], intensity curve, beat_intensity[], drops[]
VideoAnalysis  path, times[], score[], cx[], cuts[], moments[]
Moment         video_idx, time, score, scene_start, scene_end
Slot           t0, t1 (frames), intensity, kind (normal/drop)
ClipPlan       video, src_start, segments[(src_off, src_dur, speed, out_dur)],
               n_frames, transition, zoom, zoom_amt, crop_cx, beat_period, fade_in/out
EditPlan       clips[], fps, duration, music_start, music_dur
```

---

## 8. GUI design

```
┌──────────────────────────────── AMV Studio ───────────────────────────────┐
│ LEFT (scrollable settings)                  │ RIGHT (live pipeline)         │
│ ① Source: [Add videos] list · Music · Output│ ✓ Check files        ████ 100%│
│ ② Length: ☑15 ☑25 ☐30 ☑50 ☑60 ☐1:30 ☐2:00 ☐3:00 · custom · variations      │
│ ③ Style: preset · format · res · fps · look │ ● Music beats        ██░░  45%│
│    sliders: cut speed / slow-mo / zoom / FX │ ○ Find moments       ░░░░   0%│
│ ④ Options: switches + transition checkboxes │ ○ Design edit                 │
│    render quality · encoder                 │ ○ Render clips                │
│                                             │ ○ Mix & finish                │
│                                             │ Overall ████████░░░░  52%     │
│                                             │ [ log window ]                │
│                                             │ [ GENERATE ] [Cancel] [Open]  │
└─────────────────────────────────────────────┴───────────────────────────────┘
```
* Dark theme, presets (GTA Action, Hype Phonk, Cinematic Trailer, Smooth Chill) fill all controls in one click.
* Settings are saved on exit and restored on start.
* Worker thread + event queue → GUI never freezes; closing the window while running asks for confirmation and cancels cleanly.

---

## 9. Error handling matrix

| Situation | Behaviour |
|---|---|
| Missing/unreadable file, no video stream, no audio stream | `InputError` before any work, with a precise message |
| ffmpeg not found | message with install hint (`pip install imageio-ffmpeg`) |
| ffprobe missing | falls back to parsing `ffmpeg -i` output |
| Music shorter than requested length | output is shortened to the music length, warning logged |
| Video shorter than needed / few moments | moments reused with different offsets; fallback evenly spaced samples |
| Beat tracking finds nothing | synthetic 120 BPM grid + warning |
| GPU encoder fails | automatic fallback to libx264 |
| A clip render fails | retry with simplified filters → else plain clip → else black filler (timeline stays intact) |
| Concat copy fails | re-encode concat |
| Low disk space | warning < 2 GB, error < 300 MB |
| User presses Cancel | ffmpeg process killed, temp files removed, GUI returns to ready state |
| Unexpected exception | friendly dialog + traceback written to log file |

---

## 10. Performance notes
* Analysis is decode-bound; 6 fps × 192 px keeps a 1-hour 1080p source to a few minutes, and the **cache** makes repeat runs instant.
* Rendering cost ≈ clip count × short ffmpeg runs; `Fast / Balanced / High` maps to x264 `veryfast / fast / medium`.
* GPU encoders speed up rendering when available.

## 11. Testing
* `tests/make_test_media.py` – synthetic footage (moving patterns, static bars, black section, flashes) and a 120 BPM track with a drop at 16 s.
* `tests/test_effects_matrix.py` – renders every transition × zoom × slow-mo combination plus framing / look / aspect / fps / quality variants.
* `tests/test_engine.py` – end-to-end plus error handling: corrupt / missing / wrong files, cancel with cleanup, short music, short footage, multi-format output.
* `tests/test_gui_logic.py` – drives the real GUI class with a stubbed toolkit (validation, presets, full run through the event queue).

Measured on the synthetic set: tempo 120.2 BPM, drop found at 15.99 s, all cuts within 11 ms of the true kick grid, scene cuts found exactly (flashes correctly ignored), black / static footage never selected, output duration exact to the frame.

## 12. Roadmap (post v1)
Drag-and-drop, preview player, lyric/caption overlays, per-clip manual overrides, original-audio ducking, face/object-aware moments, project files, presets import/export.


---

## 12. v1.1 additions

| Area | Change |
|---|---|
| GUI smoothness | Engine runs in a spawned **process** (`worker.py`, queue + cancel Event); GUI polls every 70 ms and batches progress events; tabbed layout (CTkTabview); thread fallback if a process can't start. |
| Output | `RESOLUTIONS` 480p–2160p, fps 24/30/60, `custom_size` WxH, platform presets, `base_zoom`, framing fill / blur-fit / fit-black. |
| Effects (`effects.py`) | grain, vignette, glow (screen blend), letterbox, chroma, vhs, flicker, sharpen, trails + `fx_strength`. |
| Transitions | flash, punch, shake, glitch, whip, pan, tilt, roll, strobe, invert, rgb, dip. |
| Looks | Natural, Vivid, Cinematic, Neon, Gritty, Teal & Orange, Clean, Noir, Warm Sunset, Cold Steel. |
| Analysis | Video analysis v8: raw-feature cache with rescoring, GTA game filter (idle/menu penalty), scene-cut detection, impact curve and frame-accurate `refine_impact`. Music: downbeat phase, half/double-time grids. |
| Planner | Bar-aligned beat slots, moment diversity (MMR), action peak on a beat, slow-mo ramps starting on a beat, sparse/every/off pulse, text overlays, edit report. |
| Control | Preview with seed lock (`edit_fingerprint`), sync nudge, game audio sidechain mix, multi-song jobs, `.amvproj` projects, user presets, support bundle, UI scale. |
| Rendering | Parallel clip rendering, 3-level fallback chain, always-smooth slow-mo (blend; optional mci Motion). |
| Packaging | `build_exe.bat` (PyInstaller, untested), optional `tkinterdnd2`. |

### Selling it – things to settle first
* **Content rights**: the tool processes users' own footage/music; do not bundle or advertise GTA/Rockstar assets or names in a way implying affiliation, and market it as a generic AMV/edit maker. Check Take-Two's fan-content policy and the licences of every dependency (FFmpeg build is GPL/LGPL depending on the binary – review before redistributing).
* **Not built yet**: licence keys/activation, payment, auto-update, code signing, crash reporting, installer. 
