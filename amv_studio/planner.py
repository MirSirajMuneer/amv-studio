"""Edit planner: beat grid -> slots -> assigned moments -> speed ramps, zooms, transitions, text.

Key ideas
* Cuts land on beats; long shots start on bar lines ("downbeats").
* The action peak ("impact") of every shot is placed exactly on a beat (or right on the cut).
* Slow-motion ramps start ON a beat: fast build-up -> hit -> slow -> snap back.
* Footage is picked with a diversity penalty so the same minute of gameplay is not reused.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .audio_analysis import BeatGrid, MusicAnalysis, choose_window, make_grid
from .config import EditSettings
from .errors import PlanError
from .logger import get_logger
from .video_analysis import Moment, VideoAnalysis

log = get_logger()

HIT_PREROLL = 0.10        # seconds of build-up before a "hit on the cut"


@dataclass
class Segment:
    src_off: float    # offset from the clip's src_start (source seconds)
    src_dur: float    # source seconds consumed
    speed: float      # playback speed (1.0 normal, 0.4 slow-mo, 1.25 fast)
    out_dur: float    # seconds on the output timeline


@dataclass
class TextOverlay:
    text: str
    start: float
    end: float
    pos: str = "Top"          # Top | Center | Bottom
    kind: str = "title"       # title | outro | mark


@dataclass
class ClipPlan:
    index: int
    video_path: str
    src_fps: float
    src_start: float
    segments: List[Segment]
    n_frames: int
    out_dur: float
    transition: str = "none"
    trans_dir: int = 1             # +1 / -1: direction of pan / tilt / roll
    zoom: str = "none"             # none | push | pull | pulse | hit
    zoom_amt: float = 0.0
    crop_cx: float = 0.5
    beat_period: float = 0.5
    fade_in: float = 0.0
    fade_out: float = 0.0
    intensity: float = 0.5
    kind: str = "normal"
    slowmo: bool = False
    hit_off: Optional[float] = None    # output offset (s) at which the action peak should land
    hit_err: Optional[float] = None    # achieved minus wanted (s); None = not aligned / unknown
    has_audio: bool = False
    out_start: float = 0.0             # start time of this clip in the output (s)
    impact_src: float = 0.0
    score: float = 0.0
    video_name: str = ""

    @property
    def src_span(self) -> float:
        return max((s.src_off + s.src_dur) for s in self.segments)


@dataclass
class EditPlan:
    clips: List[ClipPlan]
    fps: int
    duration: float
    music_start: float
    music_dur: float
    seed: int
    overlays: List[TextOverlay] = field(default_factory=list)
    sync_nudge_ms: int = 0
    tempo: float = 0.0
    music_name: str = ""

    def summary(self) -> str:
        slow = sum(1 for c in self.clips if c.slowmo)
        tr: Dict[str, int] = {}
        for c in self.clips:
            tr[c.transition] = tr.get(c.transition, 0) + 1
        zooms = sum(1 for c in self.clips if c.zoom != "none")
        al = [c for c in self.clips if c.hit_err is not None]
        ok = sum(1 for c in al if abs(c.hit_err or 0) <= 0.06)
        return (f"{len(self.clips)} clips · {slow} slow-mo · {zooms} zooms · action on beat {ok}/{len(self.clips)} · "
                + ", ".join(f"{k}×{v}" for k, v in sorted(tr.items()) if k != "none"))

    def report_text(self) -> str:
        lines = [f"AMV Studio edit report — {self.music_name}",
                 f"Length {self.duration:.1f}s · {self.fps} fps · tempo {self.tempo:.0f} BPM · "
                 f"song section starts at {self.music_start:.1f}s · seed {self.seed}", self.summary(), "",
                 f"{'#':>3} {'out':>7} {'len':>5}  {'source':<22} {'src time':>8} {'score':>5}  "
                 f"{'slow':<4} {'transition':<10} {'zoom':<6} {'hit(s)':>6} {'err ms':>6}"]
        for c in self.clips:
            err = "-" if c.hit_err is None else f"{c.hit_err * 1000:+.0f}"
            hit = "-" if c.hit_off is None else f"{c.hit_off:.2f}"
            lines.append(f"{c.index + 1:>3} {c.out_start:>7.2f} {c.out_dur:>5.2f}  {c.video_name[:22]:<22} "
                         f"{c.impact_src:>8.1f} {c.score:>5.2f}  {'yes' if c.slowmo else '':<4} "
                         f"{c.transition:<10} {c.zoom:<6} {hit:>6} {err:>6}")
        return "\n".join(lines)


@dataclass
class _Slot:
    t0: float
    t1: float
    intensity: float
    kind: str
    beats: int
    beat_times: List[float]      # beat times relative to slot start (first is 0)
    f0: int = 0
    f1: int = 0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# --------------------------------------------------------------------------- slots
def _build_slots(music: MusicAnalysis, grid: BeatGrid, m_start: float, m_dur: float, fps: int,
                 s: EditSettings) -> List[_Slot]:
    beats = grid.beats
    mask = (beats >= m_start - 1e-6) & (beats < m_start + m_dur - 0.2)
    gidx = np.nonzero(mask)[0]
    t_rel = beats[mask] - m_start
    if len(t_rel) == 0 or t_rel[0] > 0.08:
        gidx = np.concatenate([[(gidx[0] - 1) if len(gidx) else 0], gidx])
        t_rel = np.concatenate([[0.0], t_rel])
    else:
        t_rel = t_rel.copy()
        t_rel[0] = 0.0
    period = grid.period
    n_b = len(t_rel)
    inten = np.array([music.intensity_at(m_start + t) for t in t_rel])
    drop_idx = set()
    for d in music.drops:
        r = d - m_start
        if 0.0 <= r < m_dur - 1.0:
            j = int(np.argmin(np.abs(t_rel - r)))
            if abs(t_rel[j] - r) < 0.35:
                drop_idx.add(j)
    base = _lerp(2.8, 0.45, s.cut_speed)
    min_slot, max_slot = float(s.min_clip), float(max(s.max_clip, s.min_clip + 0.2))
    choices = np.array([1, 2, 4, 8])
    slots: List[_Slot] = []
    i = 0
    while i < n_b:
        t0 = float(t_rel[i])
        kind = "normal"
        if i in drop_idx:
            kind, k = "drop", 2
        else:
            want = float(np.clip(base * float(np.clip(1.5 - inten[i], 0.55, 1.5)), min_slot, max_slot))
            ratio = max(want / period, 0.5)
            k = int(choices[np.argmin(np.abs(np.log(ratio / choices)))])
            while k * period < min_slot * 0.95 and k < 8:
                k *= 2
            while k > 1 and k * period > max_slot * 1.15:
                k //= 2
            if k >= 4:                       # long shots start on a bar line and end on one
                bar_pos = (int(gidx[i]) - grid.phase) % 4
                if bar_pos != 0:
                    k = max(1, min(k, 4 - bar_pos))
        j = min(i + k, n_b)
        t1 = float(t_rel[j]) if j < n_b else float(m_dur)
        if m_dur - t1 < max(min_slot * 0.9, 0.5 * period):
            t1, j = float(m_dur), n_b
        bt = [float(t_rel[x] - t_rel[i]) for x in range(i, j)]
        slots.append(_Slot(t0, t1, float(inten[i]), kind, max(1, j - i), bt))
        i = j
    if slots and slots[-1].t1 < m_dur - 1e-3:
        slots[-1].t1 = float(m_dur)
    # snap to output frames
    total = int(round(m_dur * fps))
    snapped: List[_Slot] = []
    prev = 0
    for sl in slots:
        f1 = total if sl is slots[-1] else int(round(sl.t1 * fps))
        if f1 - prev < 3 and snapped:
            continue
        sl.f0, sl.f1 = prev, f1
        snapped.append(sl)
        prev = f1
    if snapped:
        snapped[-1].f1 = total
    return snapped


# --------------------------------------------------------------------------- footage
def _diverse_order(moments: List[Moment], scores: np.ndarray, n_pick: int, diversity: float) -> List[int]:
    """Greedy pick: best score, but moments close in time (same video) to chosen ones are penalised."""
    n = len(moments)
    times = np.array([m.time for m in moments])
    vids = np.array([m.video_idx for m in moments])
    avail = np.ones(n, bool)
    pen = np.zeros(n)
    chosen: List[int] = []
    for _ in range(min(n_pick, n)):
        adj = np.where(avail, scores * (1.0 - diversity * pen), -np.inf)
        k = int(np.argmax(adj))
        chosen.append(k)
        avail[k] = False
        pen = np.maximum(pen, np.exp(-np.abs(times - times[k]) / 12.0) * (vids == vids[k]))
    rest = [int(i) for i in np.argsort(-scores) if avail[i]]
    return chosen + rest


def _place(impact: float, scene_a: float, scene_b: float, video_dur: float, span: float,
           peak_off: float) -> Tuple[float, float]:
    """Return (src_start, scale). The impact sits `peak_off` (x scale) seconds into the clip.
    scale < 1 means the scene is too short and the clip must be played slower to fit."""
    a_min = max(scene_a, 0.0)
    b_max = min(scene_b, video_dur - 0.05)
    room = b_max - a_min
    scale = 1.0
    if room < span:
        sc = room / span if span > 0 else 1.0
        if sc >= 0.55 and room > 0.3:
            scale = sc
        else:                       # scene too short: allow running across scene boundaries
            a_min, b_max = 0.0, video_dur - 0.05
            room = b_max - a_min
            if room < span:
                scale = max(0.3, room / span)
    span2 = span * scale
    start = impact - peak_off * scale
    start = min(max(start, a_min), max(a_min, b_max - span2))
    return float(max(0.0, start)), float(scale)


# --------------------------------------------------------------------------- plan
def build_plan(music: MusicAnalysis, videos: List[VideoAnalysis], settings: EditSettings,
               target_dur: float, variation: int = 0, seed: int = 0,
               refine: Optional[Callable[[str, float], float]] = None,
               progress: Optional[Callable[[float, str], None]] = None,
               impact_cache: Optional[Dict[Tuple[str, float], float]] = None) -> EditPlan:
    rng = np.random.default_rng(int(seed) + 7919 * variation)
    s = settings
    fps = s.fps
    prog = progress or (lambda f, m="": None)
    m_start, m_dur = choose_window(music, target_dur, variation, s.auto_music_section, s.music_start)
    if m_dur < 3:
        raise PlanError("The selected music section is too short.", "Choose a longer track or earlier start.")
    grid = make_grid(music, s.beat_mode)
    slots = _build_slots(music, grid, m_start, m_dur, fps, s)
    if not slots:
        raise PlanError("Could not create a beat grid from the music.")

    moments: List[Moment] = []
    for vi, va in enumerate(videos):
        va.rescore(s.game_filter)
        moments += va.moments(vi)
    if not moments:
        raise PlanError("No usable moments were found in the video(s).",
                        "Try a longer video, or switch off 'Skip menus / idle scenes'.")
    n = len(slots)
    noisy = np.array([max(1e-3, m.score) * (1.0 + rng.uniform(-0.15, 0.15)) for m in moments])
    order_m = _diverse_order(moments, noisy, int(n * 1.3) + 2, s.diversity)

    # ---- assignment ---------------------------------------------------------
    assigned: List[int] = [0] * n
    reuse: List[int] = [0] * n
    used: Dict[int, int] = {}
    if s.order_mode == "story":
        pick = sorted(order_m[:min(n, len(order_m))], key=lambda i: (moments[i].video_idx, moments[i].time))
        for si in range(n):
            mi = pick[min(len(pick) - 1, si * len(pick) // n)]
            assigned[si] = mi
            reuse[si] = used.get(mi, 0)
            used[mi] = reuse[si] + 1
    elif s.order_mode == "random":
        pool_ = order_m[:min(len(order_m), max(n, 8))]
        perm = list(rng.permutation(len(pool_)))
        for si in range(n):
            mi = pool_[perm[si % len(pool_)]]
            assigned[si] = mi
            reuse[si] = si // len(pool_)
    else:
        slot_order = sorted(range(n), key=lambda si: (slots[si].kind != "drop", -slots[si].intensity))
        for rank, si in enumerate(slot_order):
            assigned[si] = order_m[rank % len(order_m)]
            reuse[si] = rank // len(order_m)

    # ---- frame-accurate impact times (decode short windows, in parallel) ---------
    cache = impact_cache if impact_cache is not None else {}

    def focus_of(mi: int, k: int) -> float:
        m = moments[mi]
        if k == 0:
            return m.impact
        shift = ((-1) ** k) * 1.4 * ((k + 1) // 2)      # same moment used again: look at another part of it
        return float(np.clip(m.time + shift, 0.0, videos[m.video_idx].duration - 0.5))

    keys = sorted({(assigned[si], reuse[si]) for si in range(n)})
    impacts: Dict[Tuple[int, int], float] = {}
    if s.action_on_beat and refine is not None:
        def work(key: Tuple[int, int]) -> float:
            mi, k = key
            path_ = videos[moments[mi].video_idx].path
            f = focus_of(mi, k)
            ck = (path_, round(f, 2))
            if ck not in cache:
                cache[ck] = refine(path_, f)
            return cache[ck]
        done = 0
        with ThreadPoolExecutor(max_workers=4) as ex:
            for key, val in zip(keys, ex.map(work, keys)):
                impacts[key] = val
                done += 1
                prog(done / max(1, len(keys)) * 0.9, f"Locking action to the beat… {done}/{len(keys)}")
    for key in keys:
        impacts.setdefault(key, focus_of(*key))

    # ---- per-clip design --------------------------------------------------------
    clips: List[ClipPlan] = []
    last_slow = False
    last_tr = "none"
    last_pulse = False
    pulses_used = 0
    max_pulses = max(1, n // 6)
    pool = list(s.transitions)
    for si, slot in enumerate(slots):
        mi = assigned[si]
        m = moments[mi]
        va = videos[m.video_idx]
        n_frames = slot.f1 - slot.f0
        d = n_frames / fps
        period_l = max(0.2, (slot.t1 - slot.t0) / max(1, slot.beats))
        impact = impacts[(mi, reuse[si])]
        aligned_known = s.action_on_beat and refine is not None

        # ---- slow-motion structure (ramp starts on a beat)
        segs: Optional[List[Segment]] = None
        start = 0.0
        hit_out = HIT_PREROLL
        hit_err: Optional[float] = None
        do_slow = False
        if s.slowmo and d >= 0.8:
            p = s.slowmo_freq * (0.35 + 0.65 * m.score)
            do_slow = slot.kind == "drop" or ((not last_slow) and bool(rng.random() < p))
        if do_slow:
            floor = (12.0 if s.slowmo_smoothing == "Motion" else 18.0) / max(1.0, va.src_fps)
            s_slow = max(_lerp(0.60, 0.30, s.slowmo_amount) - (0.05 if slot.kind == "drop" else 0.0),
                         min(0.6, floor))
            fast = 1.25
            post_speed = 1.6 if slot.kind == "drop" else 1.25
            pre_o = slot.beat_times[1] if (slot.beats >= 3 and len(slot.beat_times) > 1
                                           and slot.beat_times[1] < d * 0.6) else 0.0
            post_o = min(max(0.25 * d, 0.5 * period_l), 0.4 * d)
            slow_o = d - pre_o - post_o
            if slow_o >= 0.25:
                pre_s, slow_s, post_s = pre_o * fast, slow_o * s_slow, post_o * post_speed
                hit_out = pre_o if pre_o > 0 else HIT_PREROLL
                peak_off = pre_s if pre_o > 0 else HIT_PREROLL * s_slow
                span = pre_s + slow_s + post_s
                start, scale = _place(impact, m.scene_start, m.scene_end, va.duration, span, peak_off)
                if scale >= 0.999:
                    segs = []
                    if pre_o > 0:
                        segs.append(Segment(0.0, pre_s, fast, pre_o))
                    segs.append(Segment(pre_s, slow_s, s_slow, slow_o))
                    segs.append(Segment(pre_s + slow_s, post_s, post_speed, post_o))
                    achieved = impact - start
                    err_src = achieved - peak_off
                    hit_err = err_src / (fast if pre_o > 0 else s_slow)
                else:
                    segs = None
            if segs is None:
                do_slow = False
        if not do_slow:
            cands = [HIT_PREROLL]
            if s.action_on_beat and slot.kind != "drop":
                later = [bt for bt in slot.beat_times[1:] if bt <= d - 0.35]
                if later and rng.random() < 0.5:
                    cands = [float(rng.choice(later)), HIT_PREROLL]
            elif not s.action_on_beat:
                cands = [0.35 * d]
            best = None
            for hc in cands:
                st_, sc_ = _place(impact, m.scene_start, m.scene_end, va.duration, d * 1.0, hc / 1.0)
                # speed == sc_ in the plain case: source offset of the impact is hc*sc_
                err = ((impact - st_) / sc_) - hc
                if best is None or abs(err) < abs(best[3]):
                    best = (hc, st_, sc_, err)
                if abs(err) <= 0.02:
                    break
            hit_out, start, scale, hit_err = best
            segs = [Segment(0.0, d * scale, scale, d)]
        last_slow = do_slow
        aligned = bool(aligned_known and s.action_on_beat)

        # ---- transition
        tr = "none"
        if si > 0 and pool:
            if slot.kind == "drop":
                tr = next((t for t in ("flash", "strobe", "punch") if t in pool), pool[0])
            else:
                prob = min(1.0, 0.25 + 0.75 * s.effects_amount) * (0.45 + 0.55 * slot.intensity)
                if rng.random() < prob:
                    opts = [t for t in pool if t != last_tr] or pool
                    tr = str(rng.choice(opts))
        last_tr = tr
        tdir = 1 if rng.random() < 0.5 else -1

        # ---- zoom
        zoom, zamt = "none", 0.0
        za = s.zoom_amount
        pulse_ok = False
        if s.pulse_mode == "every":
            pulse_ok = (slot.intensity > 0.45 and not do_slow and d >= period_l * 0.9
                        and rng.random() < 0.5 + 0.5 * za)
        elif s.pulse_mode == "sparse":
            big = slot.kind == "drop" or slot.intensity >= 0.8
            pulse_ok = (big and not do_slow and not last_pulse and pulses_used < max_pulses
                        and d >= period_l * 1.5 and rng.random() < 0.6)
        if za > 0.02 and pulse_ok:
            zoom, zamt = "pulse", (0.03 + 0.07 * za) * (0.8 if s.pulse_mode == "sparse" else 1.0)
            pulses_used += 1
        elif za > 0.02 and aligned and rng.random() < 0.30 + 0.45 * za:
            zoom, zamt = "hit", 0.05 + 0.10 * za
        elif za > 0.02 and rng.random() < 0.30 + 0.60 * za:
            zoom = "push" if rng.random() < 0.65 else "pull"
            zamt = (0.05 + 0.14 * za) * (1.3 if do_slow else 1.0)
        last_pulse = zoom == "pulse"

        cx = 0.5
        if s.smart_crop:
            raw = va.cx_between(impact - 1.0, impact + d)
            cx = float(np.clip(0.5 + 0.6 * (raw - 0.5), 0.2, 0.8))

        clips.append(ClipPlan(
            index=si, video_path=va.path, src_fps=va.src_fps, src_start=start, segments=segs or [],
            n_frames=n_frames, out_dur=d, transition=tr, trans_dir=tdir, zoom=zoom, zoom_amt=zamt,
            crop_cx=cx, beat_period=period_l, intensity=slot.intensity, kind=slot.kind, slowmo=do_slow,
            hit_off=(hit_out + (hit_err or 0.0)) if (aligned and 0.0 <= hit_out + (hit_err or 0.0) <= d - 0.12) else None,
            hit_err=hit_err if (aligned and hit_err is not None and abs(hit_err) <= 0.06) else None,
            has_audio=bool(getattr(va, "has_audio", False)), out_start=slot.f0 / fps,
            impact_src=impact, score=m.score, video_name=va.path.replace("\\", "/").split("/")[-1]))
        prog(0.9 + 0.1 * (si + 1) / n, "")

    clips[0].fade_in = min(0.35, clips[0].out_dur * 0.5)
    clips[-1].fade_out = min(0.8, clips[-1].out_dur * 0.6)
    D = sum(c.n_frames for c in clips) / fps

    overlays: List[TextOverlay] = []
    if s.title_text:
        overlays.append(TextOverlay(s.title_text, 0.25, min(D - 0.2, 0.25 + s.title_secs), s.title_pos, "title"))
    if s.outro_text and D > s.outro_secs + 1:
        overlays.append(TextOverlay(s.outro_text, D - s.outro_secs, D - 0.1, "Center", "outro"))
    if s.watermark_text:
        overlays.append(TextOverlay(s.watermark_text, 0.0, D, "Bottom", "mark"))

    plan = EditPlan(clips=clips, fps=fps, duration=D, music_start=m_start, music_dur=m_dur, seed=int(seed),
                    overlays=overlays, sync_nudge_ms=int(s.sync_nudge_ms), tempo=float(grid_tempo(grid)),
                    music_name=music.path.replace("\\", "/").split("/")[-1])
    log.info("Plan (%.0fs, var %d): %s", target_dur, variation, plan.summary())
    return plan


def grid_tempo(grid: BeatGrid) -> float:
    return 60.0 / grid.period if grid.period > 0 else 0.0
