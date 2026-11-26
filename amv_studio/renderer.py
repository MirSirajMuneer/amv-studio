"""Rendering: one ffmpeg run per clip (parallel, with retries/fallbacks), concat, mux music (+game sound)."""
from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import EditSettings, output_size
from .effects import RenderContext, build_audio_filter, build_filter, build_simple_filter
from .errors import CancelledError, FfmpegError, RenderError
from .ffmpeg_utils import (cpu_count, find_font, has_filter, pick_encoder, probe, run_ffmpeg,
                           video_encode_args)
from .logger import get_logger
from .planner import ClipPlan, EditPlan
from .progress import CancelToken, StepReporter

log = get_logger()


def make_context(plan: EditPlan, s: EditSettings, work_dir: Path) -> RenderContext:
    """Render settings + text files for the overlays (text is passed to ffmpeg through files)."""
    W, H = output_size(s)
    ctx = RenderContext(W=W, H=H, fps=plan.fps, framing=s.framing, look=s.look, smoothing=s.slowmo_smoothing,
                        quality=s.quality, fx=list(s.fx), fx_strength=s.fx_strength, base_zoom=s.base_zoom)
    if plan.overlays:
        font = find_font()
        if font is None or not has_filter("drawtext"):
            log.warning("Text overlays skipped: no usable font / drawtext filter in this ffmpeg build.")
        else:
            ctx.font = font
            ctx.overlays = list(plan.overlays)
            for i, ov in enumerate(plan.overlays):
                f = work_dir / f"txt_{i}.txt"
                f.write_text(ov.text, encoding="utf-8")
                ctx.text_files[i] = str(f.resolve())
    return ctx


def _clip_args(clip: ClipPlan, graph: str, enc_args: List[str], out: Path, threads: int,
               wav: Optional[Path] = None) -> List[str]:
    a = ["-ss", f"{max(0.0, clip.src_start):.4f}", "-t", f"{clip.src_span + 1.0:.3f}",
         "-i", clip.video_path, "-filter_complex", graph, "-map", "[vout]",
         "-an", "-sn", "-dn", "-frames:v", str(clip.n_frames)] + enc_args
    if threads:
        a += ["-threads", str(threads)]
    a.append(str(out))
    if wav is not None:
        a += ["-map", "[aout]", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(wav)]
    return a


def auto_workers(s: EditSettings, encoder: str) -> int:
    if s.workers:
        return s.workers
    if encoder != "libx264":
        return min(2, max(1, cpu_count() // 4))
    return max(1, min(4, cpu_count() // 4))


def render_clips(plan: EditPlan, s: EditSettings, work_dir: Path, cancel: Optional[CancelToken],
                 report: StepReporter, want_game_audio: bool = False
                 ) -> Tuple[List[Path], Dict[int, Path]]:
    """Render every clip (in parallel). Returns (clip mp4 files, {clip index: game-audio wav})."""
    ctx = make_context(plan, s, work_dir)
    encoder = pick_encoder(s.encoder)
    pixels = ctx.W * ctx.H
    sw_args = video_encode_args("libx264", s.quality, plan.fps, pixels)
    state = {"encoder": encoder, "args": video_encode_args(encoder, s.quality, plan.fps, pixels)}
    lock = threading.Lock()
    total = len(plan.clips)
    workers = max(1, min(auto_workers(s, encoder), total))
    threads = max(2, cpu_count() // workers) if encoder == "libx264" else 0
    log.info("Rendering %d clips at %dx%d@%d with %s, %d parallel worker(s)", total, ctx.W, ctx.H,
             plan.fps, encoder, workers)
    local = CancelToken(parent=cancel)
    fracs = [0.0] * total
    files: List[Optional[Path]] = [None] * total
    wavs: Dict[int, Path] = {}

    def update(i: int, f: float, msg: str = "") -> None:
        fracs[i] = max(fracs[i], min(1.0, f))
        report(sum(fracs) / total, msg or f"Rendering clips… {sum(1 for x in fracs if x >= 1.0)}/{total} done")

    def render_one(i: int) -> None:
        clip = plan.clips[i]
        local.check()
        out = work_dir / f"clip_{i:04d}.mp4"
        wav = work_dir / f"clip_{i:04d}.wav"
        label = f"Clip {i + 1}/{total}"
        use_audio = want_game_audio and clip.has_audio
        attempts = [("full", build_filter(clip, ctx, True), use_audio)]
        if ctx.overlays:
            attempts.append(("no-text", build_filter(clip, ctx, False), use_audio))
        if use_audio:
            attempts.append(("full-no-audio", build_filter(clip, ctx, ctx.overlays != []), False))
        attempts.append(("simple", build_simple_filter(clip, ctx), False))
        done, last_err = False, None
        for name, graph, with_audio in attempts:
            if with_audio:
                graph = graph + ";" + build_audio_filter(clip)
            with lock:
                enc_list = [state["args"]] if state["encoder"] == "libx264" else [state["args"], sw_args]
            for ea in enc_list:
                try:
                    run_ffmpeg(_clip_args(clip, graph, ea, out, threads, wav if with_audio else None),
                               total_duration=clip.out_dur, cancel=local,
                               on_progress=lambda f, i=i: update(i, f))
                    if ea is sw_args:
                        with lock:
                            if state["encoder"] != "libx264":
                                log.warning("Hardware encoder failed; switching to CPU (libx264).")
                                state["encoder"], state["args"] = "libx264", sw_args
                    done = out.exists() and out.stat().st_size > 0
                    if done and with_audio and wav.exists():
                        wavs[i] = wav
                    break
                except CancelledError:
                    raise
                except FfmpegError as e:
                    last_err = e
                    log.warning("%s: '%s' attempt failed: %s", label, name, e.message[-300:])
            if done:
                if name not in ("full",):
                    log.warning("%s rendered with reduced features (%s).", label, name)
                break
        if not done:
            log.error("%s could not be rendered (%s) - inserting black filler.", label, last_err)
            try:
                run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={ctx.W}x{ctx.H}:r={plan.fps}",
                            "-frames:v", str(clip.n_frames)] + sw_args + [str(out)], cancel=local)
            except CancelledError:
                raise
            except FfmpegError as e:
                raise RenderError("Rendering failed and the fallback also failed.", e.message[-300:])
        files[i] = out
        update(i, 1.0)

    errors: List[BaseException] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(render_one, i) for i in range(total)]
        for f in futs:
            try:
                f.result()
            except BaseException as e:  # noqa: BLE001 - collected, re-raised below
                if not isinstance(e, CancelledError) or not errors:
                    errors.append(e)
                local.cancel()          # stop the other workers quickly
    if cancel is not None and cancel.is_set():
        raise CancelledError()
    real = [e for e in errors if not isinstance(e, CancelledError)]
    if real:
        raise real[0]
    if errors:
        raise errors[0]
    report(1.0, f"Rendered {total} clips")
    return [f for f in files if f is not None], wavs


def _concat_list(files: List[Path], list_path: Path) -> None:
    lines = []
    for f in files:
        p = str(f.resolve()).replace("\\", "/").replace("'", r"'\''")
        lines.append(f"file '{p}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_game_track(plan: EditPlan, wavs: Dict[int, Path], work_dir: Path,
                      cancel: Optional[CancelToken]) -> Optional[Path]:
    """Concatenate per-clip game sound (silence where a clip has none) into one wav."""
    if not wavs:
        return None
    parts: List[Path] = []
    for i, clip in enumerate(plan.clips):
        if i in wavs and wavs[i].exists():
            parts.append(wavs[i])
        else:
            sil = work_dir / f"sil_{i:04d}.wav"
            run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{clip.out_dur:.4f}",
                        "-c:a", "pcm_s16le", str(sil)], cancel=cancel)
            parts.append(sil)
    lst = work_dir / "game_concat.txt"
    _concat_list(parts, lst)
    out = work_dir / "game_track.wav"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], cancel=cancel)
    return out


def assemble(plan: EditPlan, s: EditSettings, clip_files: List[Path], wavs: Dict[int, Path],
             music_path: str, out_path: Path, work_dir: Path, cancel: Optional[CancelToken],
             report: StepReporter) -> Path:
    list_path = work_dir / "concat.txt"
    _concat_list(clip_files, list_path)
    D = plan.duration
    fade_out = min(1.2, D * 0.25)

    # sync nudge: + = cuts later than the beat, so the music has to start earlier (or be delayed)
    nudge = plan.sync_nudge_ms / 1000.0
    ss = plan.music_start - nudge
    lead = 0.0
    if ss < 0:
        lead, ss = -ss, 0.0
    delay = f"adelay={int(lead * 1000)}:all=1," if lead > 0.001 else ""

    game = None
    if s.game_audio > 0 and wavs:
        try:
            game = _build_game_track(plan, wavs, work_dir, cancel)
        except FfmpegError as e:
            log.warning("Could not build the game-sound track (%s) - continuing without it.", e.message[-160:])

    def audio_graph(with_game: bool) -> str:
        norm = "loudnorm=I=-14:TP=-1.5:LRA=11," if s.normalize_audio else ""
        fin = (f"apad,atrim=duration={D:.4f},afade=t=in:st=0:d=0.3,"
               f"afade=t=out:st={max(0.0, D - fade_out):.3f}:d={fade_out:.3f}")
        mus = f"[1:a]{delay}aformat=sample_rates=48000:channel_layouts=stereo"
        if with_game:
            lvl = 0.15 + 0.85 * s.game_audio
            return (f"{mus},asplit=2[m1][m2];[2:a]volume={lvl:.3f}[g];"
                    "[g][m2]sidechaincompress=threshold=0.05:ratio=5:attack=10:release=300[gd];"
                    f"[m1][gd]amix=inputs=2:duration=first:normalize=0,{norm}{fin}[aout]")
        return f"{mus},{norm}{fin}[aout]"

    def args(with_game: bool, video_codec: List[str], dst: Path) -> List[str]:
        a = ["-f", "concat", "-safe", "0", "-i", str(list_path),
             "-ss", f"{ss:.4f}", "-t", f"{plan.music_dur + lead + abs(nudge) + 1:.4f}", "-i", music_path]
        if with_game and game is not None:
            a += ["-i", str(game)]
        a += ["-filter_complex", audio_graph(with_game and game is not None), "-map", "0:v:0", "-map", "[aout]"]
        return a + video_codec + ["-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
                                  "-t", f"{D:.4f}", "-movflags", "+faststart", str(dst)]

    tmp_out = out_path.with_suffix(".part.mp4")
    W, H = output_size(s)
    reenc = video_encode_args("libx264", s.quality, plan.fps, W * H)
    prog = lambda f: report(f, "Adding music…")  # noqa: E731
    try:
        tries = []
        if game is not None:
            tries.append(("music + game sound", True, ["-c:v", "copy"]))
        tries.append(("music", False, ["-c:v", "copy"]))
        tries.append(("music (re-encode)", False, reenc))
        last: Optional[FfmpegError] = None
        for name, with_game, vc in tries:
            try:
                run_ffmpeg(args(with_game, vc, tmp_out), total_duration=D, cancel=cancel, on_progress=prog)
                last = None
                break
            except FfmpegError as e:
                last = e
                log.warning("Final mix '%s' failed: %s", name, e.message[-200:])
        if last is not None:
            raise RenderError("Could not add the music to the video.", last.message[-300:])
        os.replace(tmp_out, out_path)
    finally:
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass
    try:
        info = probe(str(out_path))
        if abs(info.duration - D) > 0.5:
            log.warning("Output duration %.2fs differs from plan %.2fs", info.duration, D)
    except Exception:
        pass
    report(1.0, f"Saved {out_path.name}")
    return out_path


def unique_path(folder: Path, stem: str, ext: str = ".mp4") -> Path:
    stem = re.sub(r"[^\w\-. ]+", "_", stem).strip(" ._") or "amv"
    p = folder / f"{stem}{ext}"
    k = 2
    while p.exists():
        p = folder / f"{stem}_{k}{ext}"
        k += 1
    return p
