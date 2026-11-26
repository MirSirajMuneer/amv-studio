"""Pipeline orchestrator: validation -> music -> video -> plan -> render -> mix, with progress/cancel."""
from __future__ import annotations

import dataclasses
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

from . import config
from .audio_analysis import MusicAnalysis, analyze_music
from .config import EditSettings, output_size
from .errors import AmvError, CancelledError, InputError, MediaError
from .ffmpeg_utils import MediaInfo, find_ffmpeg, fmt_time, has_filter, probe
from .logger import attach_hook, detach_hook, get_logger
from .planner import EditPlan, build_plan
from .progress import CancelToken, EventCallback, StepReporter
from .renderer import assemble, render_clips, unique_path
from .video_analysis import VideoAnalysis, analyze_video, refine_impact

log = get_logger()

# (id, label, weight in overall progress)
STEPS: List[Tuple[str, str, float]] = [
    ("validate", "Check files", 0.02),
    ("music", "Analyze music & beats", 0.08),
    ("video", "Find the best moments", 0.25),
    ("plan", "Design the edit", 0.05),
    ("render", "Render clips", 0.50),
    ("mix", "Add music & finish", 0.10),
]

DEFAULT_OUTPUT = Path.home() / "AMV_Studio_Output"


def _label(dur: float) -> str:
    if dur >= 60 and dur % 60:
        return f"{int(dur // 60)}m{int(dur % 60):02d}s"
    return f"{int(dur // 60)}min" if dur >= 60 else f"{int(dur)}s"


class Pipeline:
    def __init__(self, settings: EditSettings, videos: List[str], music: Union[str, List[str]],
                 callback: Optional[EventCallback] = None, cancel: Optional[CancelToken] = None,
                 preview: bool = False):
        self.s = dataclasses.replace(settings).clamp()
        self.videos = [str(v) for v in videos]
        if isinstance(music, str):
            music = [music] if music else []
        self.musics = [str(m) for m in music if m]
        self.cb = callback
        self.cancel = cancel or CancelToken()
        self.preview = preview
        if preview:                       # quick low-resolution draft of the SAME edit (same seed)
            self.s.durations = [self.s.durations[0]]
            self.s.variations = 1
            self.s.resolution, self.s.custom_size, self.s.quality = "480p", "", "Fast"
            self.s.base_zoom = 1.0 if self.s.base_zoom > 1.0 else self.s.base_zoom
            self.musics = self.musics[:1]

    # ------------------------------------------------------------------ events
    def _emit(self, **ev) -> None:
        if self.cb:
            self.cb(ev)

    def _start(self, step: str, msg: str = "") -> None:
        self._emit(kind="step_start", step=step, message=msg)

    def _done(self, step: str, msg: str = "") -> None:
        self._emit(kind="step_done", step=step, message=msg)

    def _rep(self, step: str, lo: float = 0.0, hi: float = 1.0) -> StepReporter:
        return StepReporter(self.cb, step, lo, hi)

    # ------------------------------------------------------------------ validation
    def _validate(self) -> Tuple[List[MediaInfo], List[MediaInfo], Path]:
        rep = self._rep("validate")
        rep(0.0, "Looking for ffmpeg…")
        find_ffmpeg()
        if not self.videos:
            raise InputError("Please add at least one video file.")
        if not self.musics:
            raise InputError("Please choose at least one music file.")
        total = len(self.videos) + len(self.musics)
        infos: List[MediaInfo] = []
        for i, v in enumerate(self.videos):
            self.cancel.check()
            rep(i / total, f"Reading {Path(v).name}…")
            info = probe(v)
            if not info.has_video:
                raise MediaError(f"'{info.name}' has no video stream.",
                                 "Remove it from the list or choose a real video file.")
            if info.duration < 2.0:
                raise MediaError(f"'{info.name}' is too short ({info.duration:.1f}s).",
                                 "Use footage that is at least a few seconds long.")
            infos.append(info)
        minfos: List[MediaInfo] = []
        for j, m in enumerate(self.musics):
            self.cancel.check()
            rep((len(self.videos) + j) / total, f"Reading {Path(m).name}…")
            mi = probe(m)
            if not mi.has_audio:
                raise MediaError(f"'{mi.name}' has no audio.", "Choose an audio file (mp3, wav, m4a, flac…).")
            if mi.duration < 3.0:
                raise MediaError(f"The music '{mi.name}' is too short ({mi.duration:.1f}s).")
            minfos.append(mi)
        out_dir = Path(self.s.output_dir).expanduser() if self.s.output_dir else DEFAULT_OUTPUT
        if self.preview:
            out_dir = out_dir / "previews"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            probe_file = out_dir / ".amv_write_test"
            probe_file.write_text("ok")
            probe_file.unlink()
        except OSError as e:
            raise InputError(f"Cannot write to the output folder:\n{out_dir}",
                             f"Choose another folder. ({e.strerror or e})")
        free = shutil.disk_usage(out_dir).free
        W, H = output_size(self.s)
        need = int(W * H / 1e6 * 40) * 1024 ** 2 // 10 + 300 * 1024 ** 2     # rough: grows with resolution
        if free < min(need, 2 * 1024 ** 3):
            raise InputError("Not enough free disk space in the output folder.",
                             "Free some space or choose another drive.")
        if free < 2 * 1024 ** 3:
            log.warning("Low disk space: %.1f GB free.", free / 1024 ** 3)
        longest = max(self.s.durations)
        for mi in minfos:
            if mi.duration < longest - 0.25:
                log.warning("'%s' is %s long: outputs longer than that will be shortened to %s.",
                            mi.name, fmt_time(mi.duration), fmt_time(mi.duration))
        if self.s.title_text or self.s.outro_text or self.s.watermark_text:
            if not has_filter("drawtext"):
                log.warning("This ffmpeg build has no text filter - title / outro / watermark will be skipped.")
        if self.s.game_audio > 0 and not any(i.has_audio for i in infos):
            log.warning("Your video file(s) have no audio track - game sound is skipped.")
        if self.s.slowmo_smoothing == "Motion":
            log.warning("Motion-interpolated slow-mo is very slow (minutes per clip). "
                        "'Blend' is recommended for normal use.")
        rep(1.0, f"{len(infos)} video(s), {len(minfos)} song(s)")
        return infos, minfos, out_dir

    # ------------------------------------------------------------------ main
    def run(self) -> List[Path]:
        hook = attach_hook(lambda lvl, msg: self._emit(kind="log", level=lvl, message=msg))
        work: Optional[Path] = None
        t_start = time.time()
        try:
            seed = self.s.seed or random.randint(1, 999_999)
            log.info("=== %s %s%s · seed %d ===", config.APP_NAME, config.VERSION,
                     " (preview)" if self.preview else "", seed)
            self._emit(kind="seed", seed=seed)

            self._start("validate", "Checking your files…")
            infos, minfos, out_dir = self._validate()
            self._done("validate", f"{len(infos)} video(s), {len(minfos)} song(s) OK")

            # ---- music
            self._start("music", "Analyzing music…")
            musics: List[MusicAnalysis] = []
            for j, (mp, mi) in enumerate(zip(self.musics, minfos)):
                self.cancel.check()
                musics.append(analyze_music(mp, self.cancel, self._rep("music", j / len(minfos),
                                                                       (j + 1) / len(minfos)), mi.duration))
            self._done("music", ", ".join(f"{m.tempo:.0f} BPM" for m in musics)
                       + (" (estimated grid)" if any(m.synthetic_grid for m in musics) else ""))
            self.cancel.check()

            # ---- video
            self._start("video", "Scanning footage…")
            analyses: List[VideoAnalysis] = []
            n = len(self.videos)
            for i, (v, info) in enumerate(zip(self.videos, infos)):
                self.cancel.check()
                analyses.append(analyze_video(v, info, self.s.analysis_fps, self.cancel,
                                              self._rep("video", i / n, (i + 1) / n), self.s.use_cache,
                                              self.s.game_filter))
            n_m = sum(len(a.moments(0)) for a in analyses)
            self._done("video", f"{n_m} candidate moments")

            # ---- plan
            jobs = [(m, d, v) for m in range(len(musics)) for d in self.s.durations
                    for v in range(self.s.variations)]
            self._start("plan", "Designing edits…")
            plans: List[EditPlan] = []
            cache: dict = {}
            for ji, (mi, dur, var) in enumerate(jobs):
                self.cancel.check()
                rp = self._rep("plan", ji / len(jobs), (ji + 1) / len(jobs))
                plans.append(build_plan(
                    musics[mi], analyses, self.s, dur, var, seed,
                    refine=lambda p, t: refine_impact(p, t, cancel=self.cancel),
                    progress=lambda f, m="", rp=rp: rp(f, m or f"Planning {fmt_time(dur)} edit…"),
                    impact_cache=cache))
            self._rep("plan")(1.0, plans[0].summary())
            self._done("plan", f"{len(plans)} edit(s) planned")

            # ---- render + mix
            work = Path(tempfile.mkdtemp(prefix=".amv_work_", dir=out_dir))
            self._start("render", "Rendering…")
            self._start("mix", "Waiting…")
            outputs: List[Path] = []
            reports: List[str] = []
            nj = len(jobs)
            multi_song = len(musics) > 1
            for ji, ((mi, dur, var), plan) in enumerate(zip(jobs, plans)):
                self.cancel.check()
                job_dir = work / f"job{ji}"
                job_dir.mkdir()
                files, wavs = render_clips(plan, self.s, job_dir, self.cancel,
                                           self._rep("render", ji / nj, (ji + 1) / nj),
                                           want_game_audio=self.s.game_audio > 0)
                stem = f"{Path(self.videos[0]).stem}_{Path(self.musics[mi]).stem}_{_label(dur)}"
                if self.s.variations > 1:
                    stem += f"_v{var + 1}"
                if self.preview:
                    stem += "_PREVIEW"
                out_path = unique_path(out_dir, stem)
                out = assemble(plan, self.s, files, wavs, self.musics[mi], out_path, job_dir, self.cancel,
                               self._rep("mix", ji / nj, (ji + 1) / nj))
                outputs.append(out)
                rtxt = plan.report_text()
                reports.append(rtxt)
                try:
                    rdir = out_dir / "edit_reports"
                    rdir.mkdir(exist_ok=True)
                    (rdir / (out.stem + ".txt")).write_text(rtxt, encoding="utf-8")
                except OSError:
                    pass
                self._emit(kind="output", path=str(out), message=f"Saved {out.name} ({ji + 1}/{nj})",
                           preview=self.preview)
                shutil.rmtree(job_dir, ignore_errors=True)
            self._done("render", f"{sum(len(p.clips) for p in plans)} clips rendered")
            self._done("mix", f"{len(outputs)} file(s) saved")
            self._emit(kind="report", text="\n\n".join(reports))
            log.info("Finished in %s", fmt_time(time.time() - t_start))
            self._emit(kind="done", outputs=[str(o) for o in outputs], preview=self.preview, seed=seed,
                       message=f"Done in {fmt_time(time.time() - t_start)}")
            return outputs
        except CancelledError:
            log.info("Cancelled by user.")
            self._emit(kind="cancelled", message="Cancelled.")
            raise
        except AmvError as e:
            log.error("Failed: %s", e.message)
            self._emit(kind="error", message=e.user_text(), fatal=True)
            raise
        except Exception as e:  # unexpected: keep traceback in the log file
            log.exception("Unexpected error")
            self._emit(kind="error", fatal=True,
                       message=f"Unexpected error: {e}\n\nDetails were saved to the log file.")
            raise
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
            detach_hook(hook)
