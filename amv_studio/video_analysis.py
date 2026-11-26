"""Video analysis: find the exciting moments ("cool parts") of long footage.

A low-resolution frame stream is produced by ffmpeg (fast, hardware-friendly decoding) and OpenCV
computes per-sample features. The *raw features* are cached on disk (keyed by path/size/mtime), so the
scoring (incl. the game-footage filter) can be re-run instantly without decoding the video again.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .config import CACHE_DIR
from .errors import AnalysisError
from .ffmpeg_utils import MediaInfo, fmt_time, iter_raw_frames
from .logger import get_logger
from .progress import CancelToken, StepReporter

log = get_logger()

ALGO_VERSION = 8
AN_WIDTH = 192
RAW_KEYS = ("motion", "peak", "diff", "sat", "val", "con", "sharp", "fire", "cutsc", "cx")


@dataclass
class Moment:
    video_idx: int
    time: float          # time of the action peak (seconds in the source video)
    score: float         # 0..1
    scene_start: float   # bounds of the continuous scene containing the peak
    scene_end: float
    impact: float = 0.0  # coarse time of the sudden "hit" (explosion, crash, flash) near the peak


@dataclass
class VideoAnalysis:
    path: str
    duration: float
    src_fps: float
    an_fps: float
    times: np.ndarray
    score: np.ndarray                    # smoothed 0..1
    cx: np.ndarray                       # motion centroid (0..1 of frame width)
    cuts: np.ndarray                     # scene-cut times (seconds)
    impact_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))
    raw: Dict[str, np.ndarray] = field(default_factory=dict)
    cut_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    from_cache: bool = False
    game_filter: bool = True
    has_audio: bool = False

    # ------------------------------------------------------------------
    def rescore(self, game_filter: bool) -> "VideoAnalysis":
        if game_filter != self.game_filter and self.raw:
            self.score, self.impact_curve = _compute_score(self.raw, self.cut_idx, self.an_fps, game_filter)
            self.game_filter = game_filter
        return self

    def scene_bounds(self, t: float) -> tuple:
        prev = self.cuts[self.cuts <= t]
        nxt = self.cuts[self.cuts > t]
        a = float(prev.max()) if len(prev) else 0.0
        b = float(nxt.min()) if len(nxt) else self.duration
        return a, b

    def cx_between(self, t0: float, t1: float) -> float:
        if not len(self.cx):
            return 0.5
        i0 = int(np.clip(t0 * self.an_fps, 0, len(self.cx) - 1))
        i1 = int(np.clip(t1 * self.an_fps, i0 + 1, len(self.cx)))
        return float(np.median(self.cx[i0:i1]))

    def moments(self, video_idx: int, min_sep: float = 2.0, min_percentile: Optional[float] = None
                ) -> List[Moment]:
        s = self.score
        n = len(s)
        if n < 3:
            return []
        if min_percentile is None:
            min_percentile = 55.0 if self.game_filter else 40.0
        sep = max(1, int(min_sep * self.an_fps))
        thr = float(np.percentile(s, min_percentile))
        taken = np.zeros(n, bool)
        picks: List[int] = []
        for k in np.argsort(-s):
            if s[k] < thr or s[k] <= 0.02:
                break
            if taken[k]:
                continue
            picks.append(int(k))
            taken[max(0, k - sep): k + sep + 1] = True
        if len(picks) < 3 and self.duration > 6:  # fallback: evenly spaced samples
            for t in np.linspace(self.duration * 0.1, self.duration * 0.9, 6):
                picks.append(int(min(n - 1, t * self.an_fps)))
        out = []
        ic = self.impact_curve
        for k in sorted(set(picks)):
            t = float(self.times[k])
            a, b = self.scene_bounds(t)
            imp = t
            if len(ic) == n:
                lo, hi = max(0, k - int(1.2 * self.an_fps)), min(n, k + int(0.6 * self.an_fps) + 1)
                seg = ic[lo:hi]
                if len(seg) and seg.max() > 0.05:
                    imp = float(self.times[lo + int(np.argmax(seg))])
            out.append(Moment(video_idx, t, float(s[k]), a, b, imp))
        return out


# ------------------------------------------------------------------ cache
def _cache_file(path: str, fps: float) -> Path:
    p = Path(path)
    st = p.stat()
    key = hashlib.sha1(
        f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{fps:.3f}|{ALGO_VERSION}".encode()).hexdigest()[:24]
    return CACHE_DIR / "analysis" / f"{key}.npz"


def _from_raw(path: str, duration: float, src_fps: float, an_fps: float, raw: Dict[str, np.ndarray],
              cut_idx: np.ndarray, cuts: np.ndarray, game_filter: bool, from_cache: bool) -> VideoAnalysis:
    score, impact = _compute_score(raw, cut_idx, an_fps, game_filter)
    n = len(score)
    cx = uniform_filter1d(raw["cx"], size=max(1, int(1.0 * an_fps)), mode="nearest")
    return VideoAnalysis(path=path, duration=duration, src_fps=src_fps, an_fps=an_fps,
                         times=np.arange(n) / an_fps, score=score.astype(np.float32),
                         cx=cx.astype(np.float32), cuts=cuts.astype(np.float32),
                         impact_curve=impact.astype(np.float32), raw=raw, cut_idx=cut_idx,
                         from_cache=from_cache, game_filter=game_filter)


def _load_cache(f: Path, path: str, game_filter: bool) -> Optional[VideoAnalysis]:
    try:
        if not f.exists():
            return None
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        raw = {k: d[k] for k in RAW_KEYS}
        return _from_raw(path, meta["duration"], meta["src_fps"], meta["an_fps"], raw, d["cut_idx"],
                         d["cuts"], game_filter, True)
    except Exception as e:
        log.warning("Ignoring unreadable cache %s (%s)", f.name, e)
        return None


def _save_cache(f: Path, va: VideoAnalysis) -> None:
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"duration": va.duration, "src_fps": va.src_fps, "an_fps": va.an_fps})
        np.savez_compressed(f, cut_idx=va.cut_idx, cuts=va.cuts, meta=meta, **va.raw)
    except Exception as e:
        log.warning("Could not write analysis cache: %s", e)


# ------------------------------------------------------------------ scoring
def _norm(x: np.ndarray, floor: float) -> np.ndarray:
    hi = max(float(np.percentile(x, 95)), floor)
    lo = float(np.percentile(x, 5))
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _compute_score(raw: Dict[str, np.ndarray], cut_idx: np.ndarray, an_fps: float, game_filter: bool):
    """Raw features -> (smoothed 0..1 score per sample, impact-onset curve)."""
    motion = raw["motion"].astype(np.float64).copy()
    peak = raw["peak"].astype(np.float64).copy()
    diff = raw["diff"].astype(np.float64).copy()
    val, sat, con = raw["val"], raw["sat"], raw["con"]
    n = len(motion)
    cut_mask = np.zeros(n, bool)
    if len(cut_idx):
        cut_mask[cut_idx] = True
    for k in cut_idx:                      # a hard cut is not "motion"
        if k > 0:
            motion[k], peak[k], diff[k] = motion[k - 1], peak[k - 1], diff[k - 1]
    motion[0], peak[0] = motion[min(1, n - 1)], peak[min(1, n - 1)]

    flash = np.maximum(0.0, np.diff(val, prepend=val[0]))
    flash[cut_mask] = 0.0
    motion_n = _norm(motion, 0.8)
    peak_n = _norm(peak, 2.5)
    fire_n = np.sqrt(_norm(raw["fire"], 0.02))
    flash_n = _norm(flash, 0.05)
    sat_n = _norm(sat, 0.25)
    sharp_n = _norm(raw["sharp"], 1.0)
    con_n = _norm(con, 0.15)
    score = (0.34 * motion_n + 0.18 * peak_n + 0.14 * fire_n + 0.12 * flash_n
             + 0.10 * sat_n + 0.07 * sharp_n + 0.05 * con_n)
    pen = np.ones(n)
    pen[val < 0.10] *= 0.15          # black / very dark (loading, fades)
    pen[con < 0.06] *= 0.25          # flat frames (menus, solid colours)
    pen[motion_n < 0.04] *= 0.5      # nearly static

    # events (explosion / crash / muzzle flash / fast burst) present within +-1 s
    events = peak_n + flash_n + fire_n
    burst = np.clip(maximum_filter1d(events, size=max(3, int(2.0 * an_fps))) / 1.5, 0, 1)
    if game_filter:
        # Idle: pause menu, map, loading screen, standing still, long straight cruising.
        low = ((diff < 0.004) | (motion_n < 0.05)).astype(float)
        idle = uniform_filter1d(low, size=max(3, int(1.5 * an_fps)), mode="nearest") > 0.8
        pen[idle] *= 0.08
        steady = uniform_filter1d(motion_n, size=max(3, int(4 * an_fps)), mode="nearest")
        cruise = (burst < 0.30) & (steady > 0.05)
        pen[cruise] *= 0.55
        pen *= 0.6 + 0.4 * burst
    score = score * pen
    score_s = uniform_filter1d(score, size=max(1, int(1.2 * an_fps)), mode="nearest")
    p99 = float(np.percentile(score_s, 99)) or 1.0
    score_s = np.clip(score_s / (p99 + 1e-9), 0.0, 1.0)

    # impact = sudden rise of activity (a hit), used to line the action up with a beat
    act = 0.5 * motion_n + 0.5 * peak_n + flash_n + fire_n
    k3 = max(2, int(0.5 * an_fps))
    cs = np.concatenate([[0.0], np.cumsum(act)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - k3)
    base = np.where(idx > 0, (cs[idx] - cs[lo]) / np.maximum(1, idx - lo), act[0])   # mean of the previous k3 samples
    onset = np.maximum(0.0, act - base)
    onset[cut_mask] = 0.0
    return score_s, onset


# ------------------------------------------------------------------ extraction
def analyze_video(path: str, info: MediaInfo, an_fps: float = 6.0,
                  cancel: Optional[CancelToken] = None, report: Optional[StepReporter] = None,
                  use_cache: bool = True, game_filter: bool = True) -> VideoAnalysis:
    report = report or StepReporter(None, "video")
    name = Path(path).name
    cache_f = _cache_file(path, an_fps)
    if use_cache:
        cached = _load_cache(cache_f, path, game_filter)
        if cached is not None:
            cached.has_audio = info.has_audio
            report(1.0, f"{name}: loaded from cache")
            log.info("Video analysis cache hit: %s", name)
            return cached

    w = AN_WIDTH
    h = max(2, int(round(w * (info.height or 9) / (info.width or 16) / 2.0)) * 2)
    est = max(1, int(info.duration * an_fps))
    L: Dict[str, List[float]] = {k: [] for k in RAW_KEYS}
    thumbs: List[np.ndarray] = []
    xs = np.arange(w, dtype=np.float32)
    prev_gray: Optional[np.ndarray] = None
    prev_hist: Optional[np.ndarray] = None

    for k, frame in enumerate(iter_raw_frames(path, w, h, an_fps, cancel)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        L["sat"].append(float(hsv[..., 1].mean()) / 255.0)
        L["val"].append(float(hsv[..., 2].mean()) / 255.0)
        L["con"].append(float(gray.std()) / 128.0)
        L["sharp"].append(float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())))
        hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        L["fire"].append(float(((hh >= 4) & (hh <= 28) & (ss > 110) & (vv > 170)).mean()))
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        thumbs.append(cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
        if prev_gray is None:
            for key in ("motion", "peak", "diff", "cutsc"):
                L[key].append(0.0)
            L["cx"].append(0.5)
        else:
            d_val = float(cv2.absdiff(gray, prev_gray).mean()) / 255.0
            L["diff"].append(d_val)
            L["cutsc"].append(1.0 - float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)))
            if d_val < 0.003:   # identical / frozen frames: optical flow would only report noise
                L["motion"].append(0.0); L["peak"].append(0.0); L["cx"].append(0.5)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 2, 5, 1.2, 0)
                mag = cv2.magnitude(flow[..., 0], flow[..., 1])
                L["motion"].append(float(mag.mean()))
                L["peak"].append(float(np.percentile(mag, 97)))
                col = mag.sum(axis=0)
                tot = float(col.sum())
                L["cx"].append(float((col * xs).sum() / tot / w) if tot > 1e-3 * w * h else 0.5)
        prev_gray, prev_hist = gray, hist
        if k % 15 == 0:
            report(min(0.98, k / est), f"{name}: {fmt_time(k / an_fps)} / {fmt_time(info.duration)}")

    n = len(L["motion"])
    if n < 3:
        raise AnalysisError(f"Not enough video frames could be read from '{name}'.",
                            "The file may be too short or corrupted.")
    raw = {k: np.array(v, dtype=np.float32) for k, v in L.items()}
    cut_a, diff_a = raw["cutsc"].astype(np.float64), raw["diff"].astype(np.float64)
    thumb_a = np.array(thumbs)

    # ---- scene cuts (histogram jump AND pixel jump AND the change persists) ---------
    def _mad_thr(x: np.ndarray, floor: float) -> float:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) * 1.4826
        return max(floor, med + 6.0 * mad)

    cand = np.nonzero((cut_a > _mad_thr(cut_a, 0.45)) & (diff_a > _mad_thr(diff_a, 0.15)))[0]
    keep: List[int] = []
    for k in cand:
        if k < 1:
            continue
        d_cut = float(np.abs(thumb_a[k] - thumb_a[k - 1]).mean())
        after = [j for j in (k + 1, k + 2) if j < n]
        before = [j for j in (k - 2, k - 3) if j >= 0]
        if not after:
            continue
        if any(float(np.abs(thumb_a[j] - thumb_a[k - 1]).mean()) < 0.45 * d_cut for j in after):
            continue    # flash-in: the look returns to what it was (explosion / lightning)
        if any(float(np.abs(thumb_a[j] - thumb_a[k]).mean()) < 0.45 * d_cut for j in before):
            continue    # flash-out
        keep.append(int(k))
    cut_idx = np.array(keep, dtype=int)
    cuts: List[float] = []
    for k in cut_idx:
        t = (k - 0.5) / an_fps
        if not cuts or t - cuts[-1] > 0.5:
            cuts.append(float(max(0.0, t)))

    va = _from_raw(path, float(info.duration), float(info.fps or 30.0), float(an_fps), raw, cut_idx,
                   np.array(cuts, dtype=np.float32), game_filter, False)
    va.has_audio = info.has_audio
    _save_cache(cache_f, va)
    report(1.0, f"{name}: analysis complete ({len(cuts)} scene cuts)")
    log.info("Analyzed %s: %d samples, %d cuts", name, n, len(cuts))
    return va


# ------------------------------------------------------------------ frame-accurate impact
def refine_impact(path: str, t: float, win: float = 1.0, fps: float = 30.0,
                  cancel: Optional[CancelToken] = None) -> float:
    """Locate the moment of the 'hit' (sudden burst of motion / flash / fire) near time `t`
    with ~1/30 s accuracy by decoding a short window of the video. Falls back to `t`."""
    try:
        t0 = max(0.0, t - win)
        w, h = 128, 72
        prev = None
        e, v, f = [], [], []
        for fr in iter_raw_frames(path, w, h, fps, cancel, start=t0, duration=2 * win):
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            v.append(float(hsv[..., 2].mean()) / 255.0)
            f.append(float(((hsv[..., 0] >= 4) & (hsv[..., 0] <= 28) & (hsv[..., 1] > 110)
                            & (hsv[..., 2] > 170)).mean()))
            e.append(0.0 if prev is None else float(cv2.absdiff(g, prev).mean()) / 255.0)
            prev = g
        if len(e) < 8:
            return t
        e, v, f = np.array(e), np.array(v), np.array(f)
        dv = np.maximum(0.0, np.diff(v, prepend=v[0]))
        a = e / (e.max() + 1e-6) + 2.0 * dv / (dv.max() + 1e-6) * (dv.max() > 0.03) \
            + (np.maximum(0.0, np.diff(f, prepend=f[0])) / (f.max() + 1e-6)) * (f.max() > 0.01)
        a = uniform_filter1d(a, size=2, mode="nearest")
        onset = np.zeros(len(a))
        for k in range(2, len(a)):
            onset[k] = a[k] - float(a[max(0, k - 8):k].mean())
        k = int(np.argmax(onset))
        if onset[k] < 0.15:
            return t
        return float(t0 + k / fps)
    except Exception as ex:  # never let refinement break the edit
        log.debug("refine_impact failed (%s)", ex)
        return t
