"""Render every transition x zoom combination plus effects / looks / framings / sizes / smoothing / text /
audio graph, and assert that the FULL filter graph works (no fallback needed).

Usage: python tests/test_effects_matrix.py <media_dir> <work_dir>
"""
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amv_studio.config import FRAMINGS, FX, LOOKS, TRANSITIONS, resolution_for  # noqa: E402
from amv_studio.effects import RenderContext, build_audio_filter, build_filter  # noqa: E402
from amv_studio.ffmpeg_utils import find_font, find_ffmpeg, probe, run_ffmpeg, video_encode_args  # noqa: E402
from amv_studio.planner import ClipPlan, Segment, TextOverlay  # noqa: E402

media, work = Path(sys.argv[1]), Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
VIDEO = str(media / "test_gameplay.mp4")


def make_clip(transition="none", zoom="none", slow=False, fps=30, n=24, start=2.0, hit=True):
    d = n / fps
    if slow:
        segs = [Segment(0, 0.3 * d * 1.25, 1.25, 0.3 * d), Segment(0.3 * d * 1.25, 0.4 * d * 0.4, 0.4, 0.4 * d),
                Segment(0.3 * d * 1.25 + 0.4 * d * 0.4, 0.3 * d * 1.25, 1.25, 0.3 * d)]
    else:
        segs = [Segment(0, d, 1.0, d)]
    return ClipPlan(index=0, video_path=VIDEO, src_fps=30.0, src_start=start, segments=segs, n_frames=n,
                    out_dur=d, transition=transition, trans_dir=-1, zoom=zoom, zoom_amt=0.1, crop_cx=0.45,
                    beat_period=0.4, fade_in=0.2 if transition == "none" and zoom == "none" else 0,
                    slowmo=slow, hit_off=0.3 if hit else None, out_start=0.0)


def ctx(aspect="9:16", res="720p", fps=30, **kw):
    W, H = resolution_for(aspect, res)
    return RenderContext(W=W, H=H, fps=fps, **kw)


def render(clip, c, out, *, expect_size=None):
    g = build_filter(clip, c)
    args = ["-ss", f"{clip.src_start:.3f}", "-t", f"{clip.src_span + 1:.2f}", "-i", VIDEO, "-filter_complex", g,
            "-map", "[vout]", "-an", "-frames:v", str(clip.n_frames)] + \
        video_encode_args("libx264", "Fast", c.fps, c.W * c.H) + [str(out)]
    run_ffmpeg(args)
    info = probe(str(out))
    assert (info.width, info.height) == (c.W, c.H), ((info.width, info.height), (c.W, c.H))
    assert abs(info.duration - clip.n_frames / c.fps) < 0.12, (info.duration, clip.n_frames / c.fps)


font = find_font()
txt = work / "t.txt"
txt.write_text("NOV 19 • JASON + LUCIA", encoding="utf-8")
overlays = [TextOverlay("NOV 19 • JASON + LUCIA", 0.0, 2.0, "Top", "title"), TextOverlay("@channel", 0.0, 5.0, "Bottom", "mark")]

cases = []
for tr, zm, sl in itertools.product(["none"] + TRANSITIONS, ["none", "push", "pull", "pulse", "hit"], [False, True]):
    cases.append((f"{tr}/{zm}/{'slow' if sl else 'norm'}", make_clip(tr, zm, sl), ctx(look="Vivid")))
for fx in FX:
    cases.append((f"fx {fx}", make_clip("punch", "push", slow=True), ctx(fx=[fx], fx_strength=0.7)))
cases.append(("fx ALL", make_clip("flash", "hit", slow=True), ctx(fx=list(FX), fx_strength=1.0)))
for lk in LOOKS:
    cases.append((f"look {lk}", make_clip("flash", "pulse"), ctx(look=lk)))
for fr, asp, bz in itertools.product(FRAMINGS, ("9:16", "16:9", "1:1"), (1.0, 1.25)):
    cases.append((f"framing {fr} {asp} bz{bz}", make_clip("punch", "push"), ctx(aspect=asp, framing=fr, base_zoom=bz)))
for sm in ("Blend",):
    for q in ("Fast", "Balanced", "High"):
        cases.append((f"smoothing {sm} {q}", make_clip("none", "push", True), ctx(smoothing=sm, quality=q)))
cases.append(("60fps slowmo", make_clip("shake", "pulse", True, fps=60, n=48), ctx(fps=60)))
cases.append(("24fps", make_clip("glitch", "push", fps=24, n=24), ctx(fps=24)))
for res in ("480p", "1080p", "1440p"):
    cases.append((f"size {res}", make_clip("glitch", "push"), ctx(res=res, look="Cinematic")))
if font:
    cases.append(("text overlays", make_clip("flash", "none"), ctx(overlays=overlays, text_files={0: str(txt), 1: str(txt)}, font=font)))

ok = fail = 0
for i, (name, clip, c) in enumerate(cases):
    try:
        render(clip, c, work / f"c{i}.mp4")
        ok += 1
    except Exception as e:
        fail += 1
        print("FAIL", name, str(e)[-260:].replace("\n", " "))

# custom (non-preset) size and 4K graph construction
try:
    c = RenderContext(W=1000, H=1778, fps=30)
    render(make_clip("pan", "hit"), c, work / "custom.mp4")
    ok += 1
except Exception as e:
    fail += 1
    print("FAIL custom size", str(e)[-200:])

# audio graph of original game sound (slow-mo segments, atempo chain)
try:
    audio_src = media / "test_gameplay_audio.mp4"
    if audio_src.exists():
        clip = make_clip("none", "none", slow=True)
        clip.video_path = str(audio_src)
        g = build_audio_filter(clip)
        run_ffmpeg(["-ss", "2", "-t", "3", "-i", str(audio_src), "-filter_complex", g, "-map", "[aout]",
                    "-c:a", "pcm_s16le", str(work / "a.wav")])
        d = probe(str(work / "a.wav")).duration
        assert abs(d - clip.out_dur) < 0.05, d
        ok += 1
except Exception as e:
    fail += 1
    print("FAIL audio graph", str(e)[-200:])

print(f"effects matrix: {ok} ok, {fail} failed, {ok + fail} total")
sys.exit(1 if fail else 0)
