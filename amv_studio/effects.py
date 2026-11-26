"""Translate a ClipPlan into an ffmpeg filter_complex string.

Pipeline per clip:
  framing (fill / blur-fit / fit-black, footage scale) -> speed-ramp segments (+smoothing) -> concat
  -> [roll rotate] -> zoom / pulse / hit / shake / pan / tilt -> colour look -> look effects (grain, glow …)
  -> transition effect (flash, glitch, whip, strobe …) -> text overlays -> fades -> [vout]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .planner import ClipPlan, TextOverlay

LOOK_FILTERS = {
    "Natural": "unsharp=5:5:0.4:5:5:0.0",
    "Vivid": "eq=contrast=1.08:saturation=1.35:gamma=0.98,unsharp=5:5:0.6:5:5:0.0",
    "Cinematic": ("eq=contrast=1.12:saturation=1.08,"
                  "colorbalance=rs=-0.05:bs=0.07:rh=0.08:bh=-0.06,vignette=angle=PI/6"),
    "Neon": ("eq=contrast=1.15:saturation=1.65:brightness=0.02,"
             "colorbalance=bs=0.10:rm=0.05:gh=-0.03,unsharp=5:5:0.6:5:5:0.0"),
    "Gritty": "eq=contrast=1.25:saturation=0.82:gamma=0.95,unsharp=5:5:0.9:5:5:0.0,noise=alls=7:allf=t",
    "Teal & Orange": ("eq=contrast=1.10:saturation=1.18,"
                      "colorbalance=rs=-0.08:gs=-0.02:bs=0.10:rh=0.10:gh=0.02:bh=-0.10"),
    "Clean": "eq=contrast=1.05:saturation=1.10,unsharp=5:5:0.5:5:5:0.0,curves=preset=medium_contrast",
    "Noir": "hue=s=0,eq=contrast=1.28:brightness=-0.02,vignette=angle=PI/5,noise=alls=6:allf=t",
    "Warm Sunset": ("eq=saturation=1.20:contrast=1.06,"
                    "colorbalance=rs=0.10:gs=0.02:bs=-0.10:rh=0.08:bh=-0.06"),
    "Cold Steel": ("eq=saturation=0.85:contrast=1.12,"
                   "colorbalance=rs=-0.08:bs=0.12:rh=-0.05:bh=0.08"),
}

# zoom amplitude added by a transition (decays over ~0.3 s)
_TRANSITION_PUNCH = {"punch": 0.14, "flash": 0.10, "whip": 0.16, "glitch": 0.06, "rgb": 0.08,
                     "strobe": 0.06, "invert": 0.06}


@dataclass
class RenderContext:
    W: int
    H: int
    fps: int
    framing: str = "fill"
    look: str = "Vivid"
    smoothing: str = "Blend"
    quality: str = "Balanced"
    fx: List[str] = field(default_factory=list)
    fx_strength: float = 0.5
    base_zoom: float = 1.0
    overlays: List[TextOverlay] = field(default_factory=list)
    text_files: Dict[int, str] = field(default_factory=dict)   # overlay index -> text file path
    font: Optional[str] = None

    @property
    def flags(self) -> str:
        return "lanczos" if self.quality == "High" else "bicubic"


def _even(x: float) -> int:
    return int(x // 2) * 2


# ----------------------------------------------------------------------------- graph helper
class _G:
    """Builds a linear filter chain; consecutive plain filters are merged into one segment."""

    def __init__(self, start: str):
        self.parts: List[str] = []
        self.cur = start
        self.n = 0
        self.pend: List[str] = []

    def f(self, flt: str) -> None:
        self.pend.append(flt)

    def _flush(self) -> None:
        if self.pend:
            self.n += 1
            out = f"x{self.n}"
            self.parts.append(f"[{self.cur}]{','.join(self.pend)}[{out}]")
            self.cur, self.pend = out, []

    def sub(self, template: str) -> None:
        """template uses @I (input), @O (output), @A / @B (spare labels)."""
        self._flush()
        self.n += 1
        out = f"x{self.n}"
        self.parts.append(template.replace("@I", self.cur).replace("@O", out)
                          .replace("@A", f"y{self.n}a").replace("@B", f"y{self.n}b"))
        self.cur = out

    def end(self, final: str = "vout") -> str:
        self._flush()
        self.parts.append(f"[{self.cur}]null[{final}]")
        return ";".join(self.parts)


# ----------------------------------------------------------------------------- framing
def _framing(c: RenderContext, cx: float) -> str:
    W, H, bz, F = c.W, c.H, c.base_zoom, c.flags
    Ws, Hs = _even(W * bz + 1), _even(H * bz + 1)
    if c.framing == "blur_fit":
        bw, bh = max(2, W // 4), max(2, H // 4)
        return (
            "[0:v]split[bgs][fgs];"
            f"[bgs]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"boxblur=6:2,scale={W}:{H},eq=brightness=-0.08[bg];"
            f"[fgs]scale={Ws}:{Hs}:force_original_aspect_ratio=decrease:flags={F}[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[fr0]")
    if c.framing == "fit_black":
        return (f"[0:v]scale={Ws}:{Hs}:force_original_aspect_ratio=decrease:flags={F},"
                f"crop='min(iw,{W})':'min(ih,{H})',pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black[fr0]")
    return (f"[0:v]scale={Ws}:{Hs}:force_original_aspect_ratio=increase:flags={F},"
            f"crop={W}:{H}:x='max(0,min(in_w-out_w,{cx:.3f}*in_w-out_w/2))':y='(in_h-out_h)/2'[fr0]")


# ----------------------------------------------------------------------------- speed ramp segments
def _smooth_filter(c: RenderContext) -> str:
    if c.smoothing == "Motion":
        return (f"minterpolate=fps={c.fps}:mi_mode=mci:mc_mode=obmc:me_mode=bidir:me=epzs:"
                f"mb_size=16:search_param=8:vsbmc=0")
    return f"minterpolate=fps={c.fps}:mi_mode=blend"


def _segments_graph(clip: ClipPlan, c: RenderContext, smooth: bool = True) -> str:
    n = len(clip.segments)
    g = []
    if n == 1:
        g.append("[fr0]null[i0]")
    else:
        g.append("[fr0]split=%d%s" % (n, "".join(f"[i{k}]" for k in range(n))))
    for k, sg in enumerate(clip.segments):
        a, b = sg.src_off, sg.src_off + sg.src_dur
        chain = f"[i{k}]trim=start={a:.4f}:end={b:.4f},setpts=(PTS-STARTPTS)/{sg.speed:.4f}"
        # slow-mo is ALWAYS smoothed when the source has too few frames (otherwise frames repeat = choppy)
        if smooth and sg.speed < 0.999 and clip.src_fps * sg.speed < c.fps * 0.95:
            chain += "," + _smooth_filter(c)
        chain += f",fps={c.fps}[s{k}]"
        g.append(chain)
    if n == 1:
        g.append("[s0]tpad=stop_mode=clone:stop_duration=2[cat]")
    else:
        g.append("".join(f"[s{k}]" for k in range(n))
                 + f"concat=n={n}:v=1:a=0,tpad=stop_mode=clone:stop_duration=2[cat]")
    return ";".join(g)


# ----------------------------------------------------------------------------- zoom
def _zoom_expr(clip: ClipPlan) -> str:
    terms = []
    D = max(0.2, clip.out_dur)
    A = clip.zoom_amt
    if clip.zoom == "push" and A > 0:
        terms.append(f"{A:.4f}*(1-pow(1-min(t/{D:.3f},1),2))")
    elif clip.zoom == "pull" and A > 0:
        terms.append(f"{A:.4f}*(1-min(t/{D:.3f},1))")
    elif clip.zoom == "pulse" and A > 0:
        P = max(0.2, clip.beat_period)
        terms.append(f"{A:.4f}*pow(max(0,1-mod(t,{P:.4f})/{P:.4f}),3)")
    elif clip.zoom == "hit" and A > 0 and clip.hit_off is not None:
        T = max(0.0, clip.hit_off - 0.02)
        terms.append(f"{A:.4f}*gte(t,{T:.3f})*pow(max(0,1-max(0,t-{T:.3f})/0.28),2)")
    tr = clip.transition
    amp = _TRANSITION_PUNCH.get(tr)
    if amp:
        terms.append(f"{amp:.3f}*pow(max(0,1-t/0.30),2)")
    if tr == "shake":
        terms.append("0.05")
    elif tr in ("pan", "tilt"):
        terms.append("0.12*min(1,max(0,1-max(0,t-0.18)/0.22))")
    elif tr == "roll":
        terms.append("0.18*exp(-t*9)")
    return "+".join(terms)


# ----------------------------------------------------------------------------- look effects
def _apply_fx(g: _G, clip: ClipPlan, c: RenderContext) -> None:
    s = c.fx_strength
    W, H = c.W, c.H
    for name in c.fx:
        if name == "grain":
            g.f(f"noise=alls={int(6 + 16 * s)}:allf=t+u")
        elif name == "vignette":
            g.f(f"vignette=angle=PI/{5.5 - 2.0 * s:.2f}")
        elif name == "sharpen":
            g.f(f"unsharp=7:7:{0.7 + 0.9 * s:.2f}:7:7:0.0")
        elif name == "chroma":
            px = max(1, int(1 + 3 * s * W / 1080))
            g.f(f"rgbashift=rh=-{px}:bh={px}")
        elif name == "vhs":
            g.f(f"noise=alls={int(10 + 14 * s)}:allf=t")
            g.f(f"rgbashift=rh={max(1, int(2 * W / 1080))}:bh=-{max(1, int(2 * W / 1080))}")
            g.f("eq=saturation=1.2:contrast=1.05")
            g.f("gblur=sigma=0.7")
        elif name == "flicker":
            P = max(0.2, clip.beat_period)
            g.f(f"eq=brightness='{0.04 + 0.07 * s:.3f}*pow(max(0,1-mod(t,{P:.4f})/{P:.4f}),3)':eval=frame")
        elif name == "letterbox":
            frac = 0.10 if H >= W else 0.13
            g.f(f"drawbox=x=0:y=0:w=iw:h=ih*{frac}:color=black:t=fill")
            g.f(f"drawbox=x=0:y=ih*{1 - frac}:w=iw:h=ih*{frac}:color=black:t=fill")
        elif name == "trails" and clip.slowmo:
            g.f("tmix=frames=3:weights='1 1 1'")
        elif name == "glow":
            op = 0.22 + 0.30 * s
            g.sub("[@I]split[@A][@B];"
                  f"[@B]scale=iw/4:ih/4,gblur=sigma=7,scale={W}:{H},eq=brightness=-0.04[glw];"
                  f"[@A][glw]blend=all_mode=screen:all_opacity={op:.2f}[@O]")


def _transition_filters(g: _G, clip: ClipPlan, c: RenderContext) -> None:
    tr = clip.transition
    W = c.W
    if tr == "flash":
        g.f("fade=t=in:st=0:d=0.16:color=white")
    elif tr == "dip":
        g.f("fade=t=in:st=0:d=0.14:color=black")
    elif tr == "glitch":
        s1, s2 = max(4, int(W * 0.012)), max(2, int(W * 0.006))
        g.f(f"rgbashift=rh=-{s1}:bh={s1}:gv={s2}:enable='lt(t,0.07)'")
        g.f(f"rgbashift=rh={s1 // 2 + 2}:bh=-{s1 // 2 + 2}:gh=-{s2}:enable='between(t,0.10,0.17)'")
        g.f("noise=alls=35:allf=t:enable='lt(t,0.2)'")
    elif tr == "rgb":
        s1 = max(6, int(W * 0.018))
        g.f(f"rgbashift=rh=-{s1}:bh={s1}:gv={s1 // 2}:enable='lt(t,0.12)'")
        g.f(f"rgbashift=rh=-{s1 // 3}:bh={s1 // 3}:enable='between(t,0.12,0.22)'")
    elif tr == "whip":
        g.f("gblur=sigma=26:steps=2:enable='lt(t,0.10)'")
        g.f("gblur=sigma=9:steps=1:enable='between(t,0.10,0.2)'")
    elif tr == "pan":
        g.f("gblur=sigma=36:sigmaV=0.01:steps=2:enable='lt(t,0.10)'")
        g.f("gblur=sigma=12:sigmaV=0.01:steps=1:enable='between(t,0.10,0.20)'")
    elif tr == "tilt":
        g.f("gblur=sigma=0.01:sigmaV=36:steps=2:enable='lt(t,0.10)'")
        g.f("gblur=sigma=0.01:sigmaV=12:steps=1:enable='between(t,0.10,0.20)'")
    elif tr == "strobe":
        g.f("eq=brightness='0.75*lt(mod(t,0.09),0.045)':eval=frame:enable='lt(t,0.27)'")
    elif tr == "invert":
        g.f("negate=enable='lt(t,0.07)'")


# ----------------------------------------------------------------------------- text overlays
def _esc_path(p: str) -> Optional[str]:
    if "'" in p:
        return None
    return p.replace("\\", "/").replace(":", r"\:")


def _text_filters(g: _G, clip: ClipPlan, c: RenderContext) -> None:
    if not c.overlays or not c.font:
        return
    font = _esc_path(c.font)
    if font is None:
        return
    t0, t1 = clip.out_start, clip.out_start + clip.out_dur
    for i, ov in enumerate(c.overlays):
        if ov.end <= t0 or ov.start >= t1 or i not in c.text_files:
            continue
        tf = _esc_path(c.text_files[i])
        if tf is None:
            continue
        n_chars = max(4, len(ov.text))
        if ov.kind == "mark":
            size = int(max(18, c.H * 0.022))
            alpha = "0.60"
            y = "h*0.93"
        else:
            size = int(min(c.H * 0.058, c.W * 0.9 / (0.62 * n_chars)))
            size = max(24, size)
            a, b = ov.start, ov.end
            alpha = (f"min(1,max(0,(t+{t0:.3f}-{a:.3f})/0.25))*min(1,max(0,({b:.3f}-(t+{t0:.3f}))/0.25))")
            y = {"Top": "h*0.12", "Center": "(h-text_h)/2", "Bottom": "h*0.78"}.get(ov.pos, "h*0.12")
        bw = max(2, int(c.H / 330))
        g.f(f"drawtext=fontfile='{font}':textfile='{tf}':expansion=none:fontsize={size}:"
            f"fontcolor=white:borderw={bw}:bordercolor=black@0.85:shadowx=2:shadowy=2:"
            f"x=(w-text_w)/2:y={y}:alpha='{alpha}':"
            f"enable='between(t+{t0:.3f},{ov.start:.3f},{ov.end:.3f})'")


# ----------------------------------------------------------------------------- public builders
def build_filter(clip: ClipPlan, c: RenderContext, with_text: bool = True) -> str:
    """Full filter_complex for one clip. Output label: [vout]."""
    W, H = c.W, c.H
    head = _framing(c, clip.crop_cx) + ";" + _segments_graph(clip, c)
    g = _G("cat")
    tr = clip.transition

    if tr == "roll":
        g.f(f"rotate=a='{clip.trans_dir}*0.08*exp(-t*9)':ow=iw:oh=ih:c=black")
    z = _zoom_expr(clip)
    if z:
        g.f(f"scale=w='trunc({W}*(1+({z}))/2)*2':h='trunc({H}*(1+({z}))/2)*2':eval=frame:flags={c.flags}")
        x_e, y_e = "(in_w-out_w)/2", "(in_h-out_h)/2"
        if tr == "shake":
            sh = max(4, int(W * 0.014))
            x_e += f"+{sh}*exp(-t*9)*sin(t*75)"
            y_e += f"+{sh}*exp(-t*9)*cos(t*95)"
        elif tr == "pan":
            x_e = f"(in_w-out_w)/2*(1+{clip.trans_dir}*exp(-t*14))"
        elif tr == "tilt":
            y_e = f"(in_h-out_h)/2*(1+{clip.trans_dir}*exp(-t*14))"
        g.f(f"crop={W}:{H}:x='{x_e}':y='{y_e}'")

    lf = LOOK_FILTERS.get(c.look)
    if lf:
        g.f(lf)
    _apply_fx(g, clip, c)
    _transition_filters(g, clip, c)
    if with_text:
        _text_filters(g, clip, c)
    if clip.fade_in > 0:
        g.f(f"fade=t=in:st=0:d={clip.fade_in:.3f}")
    if clip.fade_out > 0:
        g.f(f"fade=t=out:st={max(0.0, clip.out_dur - clip.fade_out):.3f}:d={clip.fade_out:.3f}")
    g.f("setsar=1")
    return head + ";" + g.end("vout")


def build_simple_filter(clip: ClipPlan, c: RenderContext) -> str:
    """Minimal fallback graph (no zoom/look/effects/text) used when the full graph fails."""
    simple = RenderContext(W=c.W, H=c.H, fps=c.fps, framing=c.framing, smoothing="Blend",
                           quality="Fast", base_zoom=c.base_zoom)
    head = _framing(simple, 0.5) + ";" + _segments_graph(clip, simple)
    tail = "[cat]setsar=1"
    if clip.fade_in > 0:
        tail += f",fade=t=in:st=0:d={clip.fade_in:.3f}"
    if clip.fade_out > 0:
        tail += f",fade=t=out:st={max(0.0, clip.out_dur - clip.fade_out):.3f}:d={clip.fade_out:.3f}"
    return head + ";" + tail + "[vout]"


# ----------------------------------------------------------------------------- audio (original game sound)
def _atempo_chain(speed: float) -> str:
    parts = []
    s = speed
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


def build_audio_filter(clip: ClipPlan) -> str:
    """Filter graph producing [aout]: the clip's original sound, speed-matched, exactly out_dur long."""
    n = len(clip.segments)
    g = []
    for k, sg in enumerate(clip.segments):
        a, b = sg.src_off, sg.src_off + sg.src_dur
        chain = f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS"
        if abs(sg.speed - 1.0) > 0.01:
            chain += "," + _atempo_chain(sg.speed)
        g.append(chain + f"[a{k}]")
    cat = "".join(f"[a{k}]" for k in range(n)) + f"concat=n={n}:v=0:a=1" if n > 1 else "[a0]anull"
    D = clip.out_dur
    g.append(f"{cat},aformat=sample_rates=48000:channel_layouts=stereo,apad,atrim=duration={D:.4f},"
             f"afade=t=in:st=0:d=0.012,afade=t=out:st={max(0.0, D - 0.012):.4f}:d=0.012[aout]")
    return ";".join(g)
