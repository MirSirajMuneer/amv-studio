"""Everything that touches ffmpeg / ffprobe: discovery, probing, running, streaming, encoders."""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

from .errors import CancelledError, FfmpegError, MediaError
from .logger import get_logger
from .progress import CancelToken

log = get_logger()

_FFMPEG: Optional[str] = None
_FFPROBE: Optional[str] = None
_HW_ENCODERS: Optional[List[str]] = None


def _popen_kwargs() -> dict:
    kw: dict = {}
    if sys.platform == "win32":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW: no flashing console windows
    return kw


def _works(exe: str) -> bool:
    try:
        r = subprocess.run([exe, "-version"], capture_output=True, timeout=15, **_popen_kwargs())
        return r.returncode == 0
    except Exception:
        return False


def find_ffmpeg() -> str:
    """Locate a working ffmpeg: env var AMV_FFMPEG, system PATH, then imageio-ffmpeg's bundled binary."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    candidates: List[str] = []
    if os.environ.get("AMV_FFMPEG"):
        candidates.append(os.environ["AMV_FFMPEG"])
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        candidates.append(sys_ff)
    try:
        import imageio_ffmpeg  # type: ignore
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for c in candidates:
        if c and _works(c):
            _FFMPEG = c
            log.info("Using ffmpeg: %s", c)
            return c
    raise FfmpegError(
        "ffmpeg was not found on this computer.",
        "Run:  pip install imageio-ffmpeg   (it bundles ffmpeg), or install ffmpeg and add it to PATH.")


def find_ffprobe() -> Optional[str]:
    global _FFPROBE
    if _FFPROBE is not None:
        return _FFPROBE or None
    cands: List[str] = []
    if os.environ.get("AMV_FFPROBE"):
        cands.append(os.environ["AMV_FFPROBE"])
    p = shutil.which("ffprobe")
    if p:
        cands.append(p)
    try:
        ff = Path(find_ffmpeg())
        sib = ff.with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if sib.exists():
            cands.append(str(sib))
    except FfmpegError:
        pass
    for c in cands:
        if _works(c):
            _FFPROBE = c
            return c
    _FFPROBE = ""
    return None


# --------------------------------------------------------------------------- probing
@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False

    @property
    def name(self) -> str:
        return Path(self.path).name


def _parse_rate(txt: str) -> float:
    try:
        if "/" in txt:
            a, b = txt.split("/")
            return float(a) / float(b) if float(b) else 0.0
        return float(txt)
    except Exception:
        return 0.0


def probe(path: str) -> MediaInfo:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise MediaError(f"File not found: {path}")
    if p.stat().st_size == 0:
        raise MediaError(f"File is empty: {p.name}")
    probe_bin = find_ffprobe()
    if probe_bin:
        try:
            r = subprocess.run(
                [probe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)],
                capture_output=True, timeout=90, **_popen_kwargs())
        except subprocess.TimeoutExpired:
            raise MediaError(f"Reading '{p.name}' timed out.", "The file may be on a slow/disconnected drive.")
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            raise MediaError(f"'{p.name}' is not a readable media file ({err[0]}).",
                             "The file may be corrupted or still downloading.")
        try:
            data = json.loads(r.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            raise MediaError(f"Could not read metadata of '{p.name}'.")
        info = MediaInfo(path=str(p), duration=0.0)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and not info.has_video:
                if s.get("disposition", {}).get("attached_pic"):
                    continue  # cover art, not a real video stream
                info.has_video = True
                info.width = int(s.get("width") or 0)
                info.height = int(s.get("height") or 0)
                info.fps = _parse_rate(s.get("avg_frame_rate", "0")) or _parse_rate(s.get("r_frame_rate", "0"))
                rot = 0
                for sd in s.get("side_data_list", []) or []:
                    if "rotation" in sd:
                        rot = int(float(sd["rotation"]))
                rot = rot or int(float((s.get("tags") or {}).get("rotate", 0) or 0))
                if abs(rot) % 180 == 90:
                    info.width, info.height = info.height, info.width
            elif s.get("codec_type") == "audio":
                info.has_audio = True
        try:
            info.duration = float(data.get("format", {}).get("duration") or 0)
        except (TypeError, ValueError):
            info.duration = 0.0
        if info.duration <= 0:
            for s in data.get("streams", []):
                try:
                    info.duration = max(info.duration, float(s.get("duration") or 0))
                except (TypeError, ValueError):
                    pass
        return info
    return _probe_with_ffmpeg(p)


def _probe_with_ffmpeg(p: Path) -> MediaInfo:
    """Fallback when ffprobe is not available: parse `ffmpeg -i` stderr."""
    r = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(p)], capture_output=True,
                       timeout=90, **_popen_kwargs())
    txt = r.stderr.decode("utf-8", "replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if not m:
        raise MediaError(f"'{p.name}' is not a readable media file.",
                         "The file may be corrupted or not a video/audio file.")
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    info = MediaInfo(path=str(p), duration=dur)
    vm = re.search(r"Stream #\d+:\d+.*?Video:.*?,\s*(\d{2,5})x(\d{2,5})", txt)
    if vm and "attached pic" not in txt:
        info.has_video = True
        info.width, info.height = int(vm.group(1)), int(vm.group(2))
        fm = re.search(r"(\d+(?:\.\d+)?)\s*fps", txt)
        info.fps = float(fm.group(1)) if fm else 30.0
    info.has_audio = bool(re.search(r"Stream #\d+:\d+.*?Audio:", txt))
    return info


# --------------------------------------------------------------------------- running
def _drain_stderr(stream, sink: "collections.deque[str]") -> None:
    try:
        for raw in iter(stream.readline, b""):
            sink.append(raw.decode("utf-8", "replace").rstrip())
    except Exception:
        pass


def _watchdog(proc: subprocess.Popen, cancel: Optional[CancelToken]) -> threading.Thread:
    def run() -> None:
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                try:
                    proc.kill()
                except Exception:
                    pass
                return
            threading.Event().wait(0.15)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def run_ffmpeg(args: List[str], total_duration: Optional[float] = None,
               cancel: Optional[CancelToken] = None,
               on_progress: Optional[Callable[[float], None]] = None) -> None:
    """Run ffmpeg to completion. Raises FfmpegError (with the tail of stderr) or CancelledError."""
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats"] + [str(a) for a in args]
    log.debug("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_popen_kwargs())
    tail: "collections.deque[str]" = collections.deque(maxlen=40)
    t_err = threading.Thread(target=_drain_stderr, args=(proc.stderr, tail), daemon=True)
    t_err.start()
    wd = _watchdog(proc, cancel)
    try:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if on_progress and total_duration and line.startswith(("out_time_us=", "out_time_ms=")):
                try:
                    secs = int(line.split("=", 1)[1]) / 1_000_000.0
                    on_progress(min(1.0, max(0.0, secs / total_duration)))
                except ValueError:
                    pass
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        t_err.join(timeout=2)
        wd.join(timeout=2)
    if cancel is not None and cancel.is_set():
        raise CancelledError()
    if proc.returncode != 0:
        msg = "\n".join(list(tail)[-8:]) or f"exit code {proc.returncode}"
        raise FfmpegError(f"ffmpeg failed: {msg}")
    if on_progress:
        on_progress(1.0)


def _read_exact(stream, n: int) -> Optional[bytes]:
    data = stream.read(n)
    if data is None or len(data) < n:
        return None
    return data


def iter_raw_frames(path: str, width: int, height: int, fps: float,
                    cancel: Optional[CancelToken] = None, start: Optional[float] = None,
                    duration: Optional[float] = None) -> Iterator[np.ndarray]:
    """Yield BGR uint8 frames (height, width, 3) sampled at `fps`, downscaled by ffmpeg.
    `start` / `duration` (seconds) decode only a section of the file."""
    vf = f"fps={fps:.4f},scale={width}:{height}:flags=area"
    pre = ["-ss", f"{max(0.0, start):.3f}"] if start else []
    post = ["-t", f"{duration:.3f}"] if duration else []
    cmd = ([find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error"] + pre + ["-i", str(path)] + post
           + ["-an", "-sn", "-dn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=width * height * 3 * 4, **_popen_kwargs())
    tail: "collections.deque[str]" = collections.deque(maxlen=40)
    t_err = threading.Thread(target=_drain_stderr, args=(proc.stderr, tail), daemon=True)
    t_err.start()
    wd = _watchdog(proc, cancel)
    size = width * height * 3
    count = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = _read_exact(proc.stdout, size)
            if buf is None:
                break
            count += 1
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        t_err.join(timeout=2)
        wd.join(timeout=2)
    if cancel is not None and cancel.is_set():
        raise CancelledError()
    if count == 0:
        raise MediaError(f"Could not decode video frames from '{Path(path).name}'.",
                         "\n".join(list(tail)[-3:]) or "The codec may be unsupported.")


def decode_audio(path: str, sr: int = 22050, cancel: Optional[CancelToken] = None,
                 duration: Optional[float] = None,
                 on_progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
    """Decode any audio/video file to mono float32 PCM at `sr`."""
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
           "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_popen_kwargs())
    tail: "collections.deque[str]" = collections.deque(maxlen=40)
    t_err = threading.Thread(target=_drain_stderr, args=(proc.stderr, tail), daemon=True)
    t_err.start()
    wd = _watchdog(proc, cancel)
    chunks: List[bytes] = []
    got = 0
    expect = (duration or 0) * sr * 4
    try:
        assert proc.stdout is not None
        while True:
            b = proc.stdout.read(1 << 20)
            if not b:
                break
            chunks.append(b)
            got += len(b)
            if on_progress and expect > 0:
                on_progress(min(1.0, got / expect))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        t_err.join(timeout=2)
        wd.join(timeout=2)
    if cancel is not None and cancel.is_set():
        raise CancelledError()
    raw = b"".join(chunks)
    usable = len(raw) - (len(raw) % 4)
    if proc.returncode != 0 or usable == 0:
        raise MediaError(f"Could not decode audio from '{Path(path).name}'.",
                         "\n".join(list(tail)[-3:]) or "The file may have no audio track.")
    return np.frombuffer(raw[:usable], dtype="<f4").astype(np.float32)


# --------------------------------------------------------------------------- encoders
def detect_encoders() -> List[str]:
    """Return hardware H.264 encoders that actually work on this machine (cached)."""
    global _HW_ENCODERS
    if _HW_ENCODERS is not None:
        return _HW_ENCODERS
    found: List[str] = []
    try:
        ff = find_ffmpeg()
        r = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, timeout=20,
                           **_popen_kwargs())
        listing = r.stdout.decode("utf-8", "replace")
        for name in ("h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox"):
            if name not in listing:
                continue
            try:
                t = subprocess.run(
                    [ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                     "color=c=black:s=256x256:r=30:d=0.2", "-frames:v", "3", "-c:v", name,
                     "-f", "null", "-"],
                    capture_output=True, timeout=20, **_popen_kwargs())
                if t.returncode == 0:
                    found.append(name)
            except Exception:
                continue
    except Exception as e:  # never let detection break the app
        log.warning("Encoder detection failed: %s", e)
    _HW_ENCODERS = found
    log.info("Working hardware encoders: %s", found or "none (using CPU libx264)")
    return found


def pick_encoder(preference: str) -> str:
    if preference == "CPU":
        return "libx264"
    hw = detect_encoders()
    return hw[0] if hw else "libx264"


def video_encode_args(encoder: str, quality: str, fps: int, pixels: int) -> List[str]:
    preset = {"Fast": "veryfast", "Balanced": "fast", "High": "medium"}.get(quality, "fast")
    crf = {"Fast": "23", "Balanced": "20", "High": "17"}.get(quality, "20")
    gop = str(max(1, int(fps) * 2))
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(int(crf) + 1),
                "-b:v", "0", "-pix_fmt", "yuv420p", "-g", gop]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", crf, "-qp_p", crf,
                "-pix_fmt", "yuv420p", "-g", gop]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", str(int(crf) + 2), "-preset", "medium",
                "-pix_fmt", "nv12", "-g", gop]
    if encoder == "h264_videotoolbox":
        mbps = max(6, int(pixels / 1e6 * 6))
        return ["-c:v", "h264_videotoolbox", "-b:v", f"{mbps}M", "-pix_fmt", "yuv420p", "-g", gop]
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p", "-g", gop]


_FILTERS: Optional[str] = None


def has_filter(name: str) -> bool:
    """True if this ffmpeg build contains the given filter (cached)."""
    global _FILTERS
    if _FILTERS is None:
        try:
            r = subprocess.run([find_ffmpeg(), "-hide_banner", "-filters"], capture_output=True,
                               timeout=20, **_popen_kwargs())
            _FILTERS = r.stdout.decode("utf-8", "replace")
        except Exception:
            _FILTERS = ""
    return bool(re.search(rf"\s{re.escape(name)}\s", _FILTERS))


def find_font() -> Optional[str]:
    """Path of a bold system font usable by ffmpeg's drawtext (None if nothing is found)."""
    cands = [
        r"C:\Windows\Fonts\impact.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for c in cands:
        if Path(c).exists():
            return c
    return None


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
