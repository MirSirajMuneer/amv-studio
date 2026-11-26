"""Music analysis: tempo, beats, intensity curve, drops, best-section selection.

Pure NumPy/SciPy implementation (no librosa/numba) so it installs everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from .errors import AnalysisError
from .ffmpeg_utils import decode_audio
from .logger import get_logger
from .progress import CancelToken, StepReporter

log = get_logger()

SR = 22050
N_FFT = 1024
HOP = 256
ENV_FPS = SR / HOP  # ~86.13 frames per second


@dataclass
class MusicAnalysis:
    path: str
    duration: float
    tempo: float
    beats: np.ndarray                 # seconds
    intensity: np.ndarray             # per envelope frame, 0..1
    beat_intensity: np.ndarray        # per beat, 0..1
    drops: List[float] = field(default_factory=list)
    synthetic_grid: bool = False
    beat_strength: Optional[np.ndarray] = None   # onset strength at every beat
    downbeat_phase: int = 0                      # beat index (mod 4) that starts a bar

    def intensity_at(self, t: float) -> float:
        i = int(min(max(t * ENV_FPS, 0), len(self.intensity) - 1))
        return float(self.intensity[i])

    @property
    def beat_period(self) -> float:
        return 60.0 / self.tempo if self.tempo > 0 else 0.5


# ----------------------------------------------------------------------------- helpers
def _robust_norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _smooth(x: np.ndarray, seconds: float) -> np.ndarray:
    size = max(1, int(seconds * ENV_FPS))
    return uniform_filter1d(x.astype(np.float64), size=size, mode="nearest")


def _spectral_features(y: np.ndarray, cancel: Optional[CancelToken],
                       report: StepReporter) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (spectral flux, rms, low-band energy) per frame, computed in chunks."""
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-5:
        raise AnalysisError("The music file seems to be silent.", "Choose a different audio file.")
    y = (y * (0.9 / peak)).astype(np.float32)
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad), mode="reflect") if len(y) > pad else np.pad(y, (pad, pad))
    n_frames = 1 + (len(yp) - N_FFT) // HOP
    if n_frames < 8:
        raise AnalysisError("The music file is too short to analyze.")
    window = np.hanning(N_FFT).astype(np.float32)
    low_bins = int(150 * N_FFT / SR) + 1
    # three bands (bass / mids / highs) so kicks are not drowned out by hi-hats
    edges = [0, int(200 * N_FFT / SR) + 1, int(2000 * N_FFT / SR) + 1, N_FFT // 2 + 1]
    band_w = (1.0, 0.6, 0.35)
    bands = np.zeros((3, n_frames), np.float32)
    rms = np.zeros(n_frames, np.float32)
    low = np.zeros(n_frames, np.float32)
    prev_lm: Optional[np.ndarray] = None
    offs = np.arange(N_FFT)[None, :]
    CH = 2048
    for s in range(0, n_frames, CH):
        if cancel:
            cancel.check()
        e = min(n_frames, s + CH)
        idx = np.arange(s, e)[:, None] * HOP + offs
        mag = np.abs(np.fft.rfft(yp[idx] * window, axis=1)).astype(np.float32)
        rms[s:e] = np.sqrt(np.mean(mag ** 2, axis=1))
        low[s:e] = mag[:, :low_bins].sum(axis=1)
        lm = np.log1p(mag * 10.0)
        d = np.maximum(np.diff(lm, axis=0, prepend=(prev_lm if prev_lm is not None else lm[:1])), 0)
        for b in range(3):
            bands[b, s:e] = d[:, edges[b]:edges[b + 1]].sum(axis=1)
        prev_lm = lm[-1:]
        report(s / n_frames * 0.7 + 0.05, "")
    flux = np.zeros(n_frames, np.float32)
    for b in range(3):
        flux += band_w[b] * bands[b] / (float(bands[b].mean()) + 1e-9)
    return flux, rms, low


def _estimate_tempo(env: np.ndarray, fps: float, lo: float = 60.0, hi: float = 200.0,
                    start_bpm: float = 120.0) -> float:
    x = env - env.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (n - np.arange(n))
    min_lag = max(2, int(fps * 60.0 / hi))
    max_lag = min(n // 2 - 1, int(fps * 60.0 / lo))
    if max_lag <= min_lag + 2:
        return start_bpm
    lags = np.arange(min_lag, max_lag + 1)
    score = ac[lags].copy()
    two = lags * 2
    ok = two < n
    score[ok] += 0.5 * ac[two[ok]]
    half = lags // 2
    score += 0.25 * ac[half]
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / start_bpm) / 1.0) ** 2)
    score = score * prior
    k = int(np.argmax(score))
    # parabolic interpolation for sub-frame accuracy
    if 0 < k < len(lags) - 1:
        a, b, c = score[k - 1], score[k], score[k + 1]
        denom = (a - 2 * b + c)
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    else:
        shift = 0.0
    lag = lags[k] + float(np.clip(shift, -1, 1))
    return float(60.0 * fps / lag)


def _track_beats(env: np.ndarray, fps: float, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Ellis (2007) dynamic-programming beat tracker. Returns beat frame indices."""
    period = 60.0 * fps / bpm
    wlen = max(1, int(round(period)))
    win = np.exp(-0.5 * (np.arange(-wlen, wlen + 1) * 32.0 / period) ** 2)
    local = np.convolve(env, win, mode="same")
    n = len(local)
    lo = int(round(-2 * period))
    hi = -int(round(period / 2))
    offsets = np.arange(lo, hi + 1)
    txwt = -tightness * (np.log(-offsets / period)) ** 2
    cum = np.zeros(n)
    back = np.full(n, -1, dtype=np.int64)
    first = True
    thresh = 0.01 * local.max()
    for i in range(n):
        idx = i + offsets
        prev = np.where(idx >= 0, cum[np.clip(idx, 0, n - 1)], 0.0)
        cand = txwt + prev
        j = int(np.argmax(cand))
        cum[i] = local[i] + cand[j]
        if first and local[i] < thresh:
            back[i] = -1
        else:
            back[i] = idx[j] if idx[j] >= 0 else -1
            first = False
    # last beat: local maxima of cumulative score
    mx = np.zeros(n, bool)
    mx[1:-1] = (cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:])
    if not mx.any():
        return np.array([], dtype=np.int64)
    med = np.median(cum[mx])
    cands = np.nonzero((cum * mx * 2) > med)[0]
    last = int(cands.max()) if len(cands) else int(np.argmax(cum))
    beats = [last]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    beats = np.array(beats[::-1], dtype=np.int64)
    # trim weak leading / trailing beats (as librosa does)
    if len(beats) > 6:
        sm = np.convolve(local[beats], np.hanning(5), mode="same")
        th = 0.5 * np.sqrt(np.mean(sm ** 2))
        s = 0
        while s < len(beats) - 1 and sm[s] <= th:
            s += 1
        e = len(beats)
        while e > s + 1 and sm[e - 1] <= th:
            e -= 1
        beats = beats[s:e]
    return beats


def _detect_drops(low: np.ndarray, rms: np.ndarray, beats_t: np.ndarray) -> List[float]:
    e = _smooth(0.5 * _robust_norm(low) + 0.5 * _robust_norm(rms), 0.25)
    n = len(e)
    a, b = int(4 * ENV_FPS), int(2 * ENV_FPS)
    if n < a + b + 10:
        return []
    cs = np.concatenate([[0.0], np.cumsum(e)])
    f = np.arange(a, n - b)
    before = (cs[f] - cs[f - a]) / a
    after = (cs[f + b] - cs[f]) / b
    score = after - before
    order = np.argsort(-score)
    picked: List[int] = []
    for k in order:
        if score[k] < 0.22:
            break
        if all(abs(f[k] - p) > 8 * ENV_FPS for p in picked):
            picked.append(int(f[k]))
        if len(picked) >= 6:
            break
    drops = []
    for p in sorted(picked):
        t = p / ENV_FPS
        if len(beats_t):
            nb = beats_t[np.argmin(np.abs(beats_t - t))]
            if abs(nb - t) < 0.6:
                t = float(nb)
        drops.append(float(t))
    return drops


# ----------------------------------------------------------------------------- public API
def analyze_music(path: str, cancel: Optional[CancelToken] = None,
                  report: Optional[StepReporter] = None, duration_hint: Optional[float] = None
                  ) -> MusicAnalysis:
    report = report or StepReporter(None, "music")
    report(0.0, "Decoding audio…")
    y = decode_audio(path, SR, cancel=cancel, duration=duration_hint,
                     on_progress=lambda f: report(f * 0.05, "Decoding audio…"))
    duration = len(y) / SR
    if duration < 3:
        raise AnalysisError("The music is shorter than 3 seconds.", "Choose a longer track.")
    report(0.05, "Analyzing spectrum…")
    flux, rms, low = _spectral_features(y, cancel, report)
    report(0.75, "Finding tempo…")
    # onset envelope, locally normalised
    env = flux.astype(np.float64)
    env = env - uniform_filter1d(env, size=int(ENV_FPS * 1.0), mode="nearest")
    env = np.maximum(env, 0)
    env = env / (env.std() + 1e-9)

    synthetic = False
    try:
        tempo = _estimate_tempo(env, ENV_FPS)
        report(0.80, f"Tracking beats ({tempo:.0f} BPM)…")
        frames = _track_beats(env, ENV_FPS, tempo)
    except Exception as e:  # defensive: never fail the job because of beat tracking
        log.warning("Beat tracking failed (%s); using a synthetic grid.", e)
        frames = np.array([], dtype=np.int64)
        tempo = 120.0
    if len(frames) < 4:
        synthetic = True
        tempo = tempo if 60 <= tempo <= 200 else 120.0
        beats_t = np.arange(0.0, duration, 60.0 / tempo)
        log.warning("Not enough beats detected - using a regular %.0f BPM grid.", tempo)
    else:
        # global phase correction: median offset to the nearest strong onset
        rad = int(0.06 * ENV_FPS)
        offs = []
        for b in frames:
            lo_, hi_ = max(0, b - rad), min(len(env), b + rad + 1)
            offs.append(int(np.argmax(env[lo_:hi_])) + lo_ - b)
        shift = int(np.median(offs))
        frames = frames + shift
        frames = frames[(frames >= 0) & (frames < len(env))]
        beats_t = frames / ENV_FPS
        # recompute tempo from the tracked beats (median interval) for a stable beat period
        if len(beats_t) > 8:
            iv = np.median(np.diff(beats_t))
            if iv > 0:
                tempo = float(60.0 / iv)
    report(0.90, "Measuring intensity & drops…")
    inten = 0.6 * _robust_norm(_smooth(rms.astype(np.float64), 0.5)) \
        + 0.4 * _robust_norm(_smooth(env, 1.0))
    inten = np.clip(_smooth(inten, 0.5), 0, 1)
    bi_idx = np.clip((beats_t * ENV_FPS).astype(int), 0, len(inten) - 1)
    beat_int = inten[bi_idx]
    drops = _detect_drops(low, rms, beats_t)
    # beat strength (onset + bass rise) and the most likely bar start ("downbeat")
    lowrise = np.maximum(np.diff(_robust_norm(low.astype(np.float64)), prepend=0.0), 0)
    strength = (_robust_norm(env)[bi_idx] + 0.7 * _robust_norm(lowrise)[bi_idx]).astype(np.float32)
    phase = 0 if synthetic else _best_phase(strength)
    report(1.0, f"{tempo:.0f} BPM · {len(beats_t)} beats · {len(drops)} drop(s)")
    log.info("Music: %.1fs tempo=%.1f beats=%d drops=%s synthetic=%s bar-phase=%d",
             duration, tempo, len(beats_t), [round(d, 1) for d in drops], synthetic, phase)
    return MusicAnalysis(path=path, duration=duration, tempo=tempo, beats=beats_t,
                         intensity=inten.astype(np.float32), beat_intensity=beat_int,
                         drops=drops, synthetic_grid=synthetic, beat_strength=strength,
                         downbeat_phase=phase)


def _best_phase(strength: np.ndarray, bar: int = 4) -> int:
    if len(strength) < bar * 3:
        return 0
    means = [float(strength[p::bar].mean()) for p in range(bar)]
    return int(np.argmax(means))


@dataclass
class BeatGrid:
    beats: np.ndarray      # seconds
    strength: np.ndarray
    phase: int             # index (mod 4) of the bar start
    period: float          # median beat interval (seconds)


def make_grid(m: MusicAnalysis, mode: str = "Normal") -> BeatGrid:
    """Beat grid used for cutting. 'Half-time' keeps every 2nd beat, 'Double-time' adds off-beats."""
    beats = np.asarray(m.beats, dtype=np.float64)
    st = m.beat_strength if m.beat_strength is not None and len(m.beat_strength) == len(beats) \
        else np.ones(len(beats), np.float32)
    phase = int(m.downbeat_phase)
    if mode == "Half-time" and len(beats) >= 8:
        start = phase % 2
        b, s = beats[start::2], st[start::2]
        ph = _best_phase(s)
    elif mode == "Double-time" and len(beats) >= 2:
        mid = (beats[:-1] + beats[1:]) / 2.0
        b = np.empty(len(beats) + len(mid))
        b[0::2], b[1::2] = beats, mid
        s = np.empty(len(b), np.float32)
        s[0::2], s[1::2] = st, 0.3 * (st[:-1] + st[1:]) / 2.0
        ph = (2 * phase) % 4
    else:
        b, s, ph = beats, st, phase
    period = float(np.median(np.diff(b))) if len(b) > 2 else m.beat_period
    return BeatGrid(beats=b, strength=np.asarray(s), phase=ph, period=max(0.15, period))


def choose_window(m: MusicAnalysis, dur: float, variation: int = 0, auto: bool = True,
                  start_hint: float = 0.0) -> Tuple[float, float]:
    """Pick the (start, duration) of the song section to use. Start is beat-aligned."""
    L = m.duration
    if dur >= L - 0.25:
        return 0.0, float(L)
    beats = m.beats if len(m.beats) else np.arange(0, L, 0.5)
    if not auto:
        s = float(min(max(start_hint, 0.0), L - dur))
        nb = beats[np.argmin(np.abs(beats - s))]
        if nb + dur <= L:
            s = float(nb)
        return s, float(dur)
    cand = [i for i, b in enumerate(beats) if b + dur <= L + 0.05]
    if not cand:
        return 0.0, float(min(dur, L))
    cs = np.concatenate([[0.0], np.cumsum(m.intensity.astype(np.float64))])
    scored = []
    for i in cand:
        b = float(beats[i])
        a = int(b * ENV_FPS)
        z = min(len(m.intensity), int((b + dur) * ENV_FPS))
        if z <= a:
            continue
        sc = (cs[z] - cs[a]) / (z - a)
        if any(b + 0.5 <= d <= b + 0.7 * dur for d in m.drops):
            sc += 0.15
        if i % 4 == 0:
            sc += 0.02
        scored.append((sc, b))
    if not scored:
        return 0.0, float(min(dur, L))
    scored.sort(reverse=True)
    chosen: List[Tuple[float, float]] = []
    for sc, b in scored:
        if all(abs(b - c[1]) > dur * 0.6 for c in chosen):
            chosen.append((sc, b))
        if len(chosen) > variation + 2:
            break
    sc, b = chosen[variation % len(chosen)]
    return float(b), float(dur)
