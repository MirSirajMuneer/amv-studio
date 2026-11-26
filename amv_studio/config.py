"""Settings, presets and constants."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logger import APP_DIR

APP_NAME = "AMV Studio"
VERSION = "1.1.0"
SETTINGS_FILE = APP_DIR / "settings.json"
CACHE_DIR = APP_DIR / "cache"

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".ts", ".mts", ".m2ts")
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma")

DURATION_CHOICES: List[Tuple[str, float]] = [
    ("15s", 15), ("25s", 25), ("30s", 30), ("50s", 50),
    ("60s", 60), ("1:30", 90), ("2:00", 120), ("3:00", 180),
]
ASPECTS = ["9:16", "16:9", "1:1"]
RESOLUTIONS = ["480p", "720p", "1080p", "1440p", "2160p"]
RESOLUTION_LABELS = {"480p": "480p (preview)", "720p": "720p HD", "1080p": "1080p Full HD",
                     "1440p": "1440p 2K", "2160p": "2160p 4K"}
FPS_CHOICES = [24, 30, 60]
LOOKS = ["Natural", "Vivid", "Cinematic", "Neon", "Gritty", "Teal & Orange", "Clean", "Noir", "Warm Sunset", "Cold Steel"]
TRANSITIONS = ["flash", "punch", "shake", "glitch", "whip", "pan", "tilt", "roll", "strobe", "invert", "rgb", "dip"]
TRANSITION_LABELS = {"flash": "Flash", "punch": "Zoom punch", "shake": "Shake", "glitch": "Glitch",
                     "whip": "Blur zoom", "pan": "Whip pan", "tilt": "Whip tilt", "roll": "Roll",
                     "strobe": "Strobe", "invert": "Invert flash", "rgb": "RGB split", "dip": "Dip to black"}
QUALITIES = ["Fast", "Balanced", "High"]
ENCODERS = ["Auto", "CPU"]
PULSE_MODES = ["sparse", "every", "off"]
PULSE_LABELS = {"sparse": "Sparse (drop & big hits)", "every": "Every beat", "off": "Off"}
BEAT_MODES = ["Normal", "Half-time", "Double-time"]
SMOOTHING = ["Blend", "Motion"]
SMOOTHING_LABELS = {"Blend": "Blend (fast, smooth)", "Motion": "Motion interpolation (very slow)"}
FRAMINGS = ["fill", "blur_fit", "fit_black"]
FRAMING_LABELS = {"fill": "Fill (crop to frame)", "blur_fit": "Fit + blurred background", "fit_black": "Fit + black bars"}
ORDER_MODES = ["energy", "story", "random"]
ORDER_LABELS = {"energy": "Best moments on loudest beats", "story": "Story order (chronological)", "random": "Shuffle"}
# Look effects (stackable, applied to every clip). Strength is set by one slider.
FX = ["grain", "vignette", "glow", "letterbox", "chroma", "vhs", "flicker", "sharpen", "trails"]
FX_LABELS = {"grain": "Film grain", "vignette": "Vignette", "glow": "Glow / bloom", "letterbox": "Cinema bars",
             "chroma": "Chromatic aberration", "vhs": "VHS retro", "flicker": "Beat flicker",
             "sharpen": "Extra sharpen", "trails": "Ghost trails (slow-mo)"}
TEXT_POSITIONS = ["Top", "Center", "Bottom"]

# Style presets: name -> overrides applied on top of the current settings.
PRESETS: Dict[str, dict] = {
    "GTA Action (Recommended)": dict(
        cut_speed=0.60, slowmo_amount=0.45, zoom_amount=0.50, effects_amount=0.50, look="Vivid",
        slowmo=True, pulse_mode="sparse", transitions=["flash", "punch", "whip", "pan", "shake", "rgb"],
        order_mode="energy", fx=["vignette", "sharpen"], fx_strength=0.5),
    "Hype Phonk": dict(
        cut_speed=0.92, slowmo_amount=0.65, zoom_amount=0.90, effects_amount=0.90, look="Neon",
        slowmo=True, pulse_mode="every", transitions=list(TRANSITIONS), order_mode="energy",
        fx=["chroma", "flicker", "glow"], fx_strength=0.6),
    "Cinematic Trailer": dict(
        cut_speed=0.30, slowmo_amount=0.60, zoom_amount=0.40, effects_amount=0.35, look="Cinematic",
        slowmo=True, pulse_mode="off", transitions=["flash", "punch", "pan", "dip"], order_mode="story",
        fx=["grain", "vignette", "letterbox", "glow"], fx_strength=0.45),
    "Smooth Chill": dict(
        cut_speed=0.18, slowmo_amount=0.50, zoom_amount=0.30, effects_amount=0.25, look="Clean",
        slowmo=True, pulse_mode="off", transitions=["punch", "whip", "dip"], order_mode="story", fx=["glow", "vignette"], fx_strength=0.35),
    "Custom": dict(),
}

# Platform presets: format only (vertical platforms: keep captions out of the top/bottom ~15 %).
PLATFORMS: Dict[str, dict] = {
    "None": dict(),
    "TikTok": dict(aspect="9:16", resolution="1080p", fps=30),
    "YouTube Shorts": dict(aspect="9:16", resolution="1080p", fps=30),
    "Instagram Reels": dict(aspect="9:16", resolution="1080p", fps=30),
    "YouTube (16:9)": dict(aspect="16:9", resolution="1080p", fps=30),
    "YouTube 60 fps (16:9)": dict(aspect="16:9", resolution="1080p", fps=60),
}


def resolution_for(aspect: str, res: str) -> Tuple[int, int]:
    """Return (width, height), both even."""
    short, long_ = {"480p": (480, 854), "720p": (720, 1280), "1440p": (1440, 2560),
                    "2160p": (2160, 3840)}.get(res, (1080, 1920))
    if aspect == "9:16":
        return short, long_
    if aspect == "1:1":
        return short, short
    return long_, short


def parse_size(text: str) -> Optional[Tuple[int, int]]:
    """'1440x2560' / '1440 x 2560' -> (1440, 2560) (made even); None if invalid."""
    import re
    m = re.fullmatch(r"\s*(\d{3,5})\s*[xX×*]\s*(\d{3,5})\s*", text or "")
    if not m:
        return None
    w, h = int(m.group(1)) // 2 * 2, int(m.group(2)) // 2 * 2
    if not (240 <= w <= 7680 and 240 <= h <= 7680):
        return None
    return w, h


def output_size(s: "EditSettings") -> Tuple[int, int]:
    """Final pixel size: a valid custom size wins, otherwise format + resolution."""
    cs = parse_size(s.custom_size)
    return cs if cs else resolution_for(s.aspect, s.resolution)


# Keys that change the *edit itself* (not just how it is encoded). Used to decide whether a
# preview and a final render are "the same edit".
RENDER_ONLY_KEYS = {"resolution", "custom_size", "quality", "encoder", "output_dir", "workers", "seed",
                    "analysis_fps", "use_cache", "platform", "preset", "ui_scale", "slowmo_smoothing"}


@dataclass
class EditSettings:
    # Output
    durations: List[float] = field(default_factory=lambda: [30.0])
    variations: int = 1
    aspect: str = "9:16"
    resolution: str = "1080p"
    fps: int = 30
    custom_size: str = ""            # "WxH" overrides format + resolution (e.g. "1440x2560")
    framing: str = "fill"            # fill | blur_fit | fit_black
    base_zoom: float = 1.0           # footage scale inside the frame (1.0 - 1.4)
    look: str = "Vivid"
    platform: str = "None"
    # Style
    preset: str = "GTA Action (Recommended)"
    cut_speed: float = 0.60          # 0 calm .. 1 frantic
    slowmo_amount: float = 0.45
    zoom_amount: float = 0.50
    effects_amount: float = 0.50
    slowmo: bool = True
    slowmo_freq: float = 0.5         # how often slow-motion is used
    slowmo_smoothing: str = "Blend"  # "Blend" (fast, smooth) | "Motion" (motion interpolation, slow)
    pulse_mode: str = "sparse"       # "sparse" | "every" | "off"
    transitions: List[str] = field(default_factory=lambda: ["flash", "punch", "whip", "pan", "shake"])
    smart_crop: bool = True
    order_mode: str = "energy"       # "energy" | "story"
    action_on_beat: bool = True      # place the action peak exactly on a beat
    game_filter: bool = True         # penalise menus / maps / loading / idle driving & walking
    diversity: float = 0.6           # 0 = best moments only, 1 = spread across the footage
    min_clip: float = 0.35           # shortest shot (s)
    max_clip: float = 6.0            # longest shot (s)
    fx: List[str] = field(default_factory=lambda: ["vignette", "sharpen"])
    fx_strength: float = 0.5
    # Music / sync
    auto_music_section: bool = True
    music_start: float = 0.0
    normalize_audio: bool = True
    beat_mode: str = "Normal"        # Normal | Half-time | Double-time
    sync_nudge_ms: int = 0           # + = cuts later than the beat, - = earlier
    game_audio: float = 0.0          # 0 = off; level of the original game sound under the music
    # Text
    title_text: str = ""
    title_pos: str = "Top"
    title_secs: float = 2.5
    outro_text: str = ""
    outro_secs: float = 2.0
    watermark_text: str = ""
    # Engine
    quality: str = "Balanced"
    encoder: str = "Auto"
    workers: int = 0                 # 0 = auto (parallel clip rendering)
    seed: int = 0                    # 0 = random
    analysis_fps: float = 6.0
    use_cache: bool = True
    ui_scale: float = 1.0            # interface size (GUI only)
    # Paths
    output_dir: str = ""

    # ---- helpers -------------------------------------------------
    def apply_preset(self, name: str) -> None:
        for k, v in PRESETS.get(name, {}).items():
            setattr(self, k, list(v) if isinstance(v, list) else v)
        self.preset = name

    def apply_platform(self, name: str) -> None:
        for k, v in PLATFORMS.get(name, {}).items():
            setattr(self, k, v)
        self.platform = name

    def clamp(self) -> "EditSettings":
        for n in ("cut_speed", "slowmo_amount", "zoom_amount", "effects_amount", "diversity", "game_audio"):
            setattr(self, n, min(1.0, max(0.0, float(getattr(self, n)))))
        self.durations = sorted({float(d) for d in self.durations if 3 <= float(d) <= 600}) or [30.0]
        self.variations = int(min(5, max(1, self.variations)))
        if self.aspect not in ASPECTS:
            self.aspect = "9:16"
        if self.resolution not in RESOLUTIONS:
            self.resolution = "1080p"
        if int(self.fps) not in FPS_CHOICES:
            self.fps = 30
        self.fps = int(self.fps)
        if self.look not in LOOKS:
            self.look = "Natural"
        if self.quality not in QUALITIES:
            self.quality = "Balanced"
        if self.encoder not in ENCODERS:
            self.encoder = "Auto"
        if self.framing not in FRAMINGS:
            self.framing = "fill"
        if self.order_mode not in ORDER_MODES:
            self.order_mode = "energy"
        self.base_zoom = min(1.4, max(1.0, float(self.base_zoom)))
        self.slowmo_freq = min(1.0, max(0.0, float(self.slowmo_freq)))
        self.fx_strength = min(1.0, max(0.0, float(self.fx_strength)))
        self.min_clip = min(3.0, max(0.2, float(self.min_clip)))
        self.max_clip = min(20.0, max(self.min_clip + 0.3, float(self.max_clip)))
        self.fx = [f for f in self.fx if f in FX]
        self.ui_scale = min(1.4, max(0.8, float(self.ui_scale)))
        if self.custom_size and parse_size(self.custom_size) is None:
            self.custom_size = ""
        if self.pulse_mode not in PULSE_MODES:
            self.pulse_mode = "sparse"
        if self.beat_mode not in BEAT_MODES:
            self.beat_mode = "Normal"
        if self.slowmo_smoothing not in SMOOTHING:
            self.slowmo_smoothing = "Blend"
        if self.title_pos not in TEXT_POSITIONS:
            self.title_pos = "Top"
        if self.platform not in PLATFORMS:
            self.platform = "None"
        self.transitions = [t for t in self.transitions if t in TRANSITIONS]
        self.analysis_fps = min(12.0, max(2.0, float(self.analysis_fps)))
        self.music_start = max(0.0, float(self.music_start))
        self.sync_nudge_ms = int(min(150, max(-150, int(self.sync_nudge_ms))))
        self.title_secs = min(8.0, max(0.5, float(self.title_secs)))
        self.outro_secs = min(8.0, max(0.5, float(self.outro_secs)))
        self.workers = int(min(6, max(0, int(self.workers))))
        for n in ("title_text", "outro_text", "watermark_text"):
            setattr(self, n, str(getattr(self, n) or "").strip()[:120])
        return self

    def edit_fingerprint(self) -> str:
        """Stable string of everything that changes the edit (used to match preview <-> final)."""
        d = {k: v for k, v in asdict(self).items() if k not in RENDER_ONLY_KEYS}
        return json.dumps(d, sort_keys=True)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "EditSettings":
        data = dict(data)
        if "beat_pulse" in data and "pulse_mode" not in data:      # settings saved by v1.0
            data["pulse_mode"] = "sparse" if data["beat_pulse"] else "off"
        valid = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in data.items() if k in valid})
        return s.clamp()


def load_settings() -> EditSettings:
    try:
        return EditSettings.from_dict(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        return EditSettings().clamp()


def save_settings(s: EditSettings) -> None:
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(s.to_json(), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------- projects (save / load)
def save_project(path: str, settings: EditSettings, videos: List[str], musics: List[str],
                 seed: int = 0) -> None:
    data = {"app": APP_NAME, "version": VERSION, "settings": json.loads(settings.to_json()),
            "videos": list(videos), "musics": list(musics), "seed": int(seed)}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_project(path: str) -> Tuple[EditSettings, List[str], List[str], int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "settings" not in data:
        raise ValueError("This is not an AMV Studio project file.")
    s = EditSettings.from_dict(data["settings"])
    musics = data.get("musics")
    if musics is None and data.get("music"):
        musics = [data["music"]]
    return s, list(data.get("videos", [])), list(musics or []), int(data.get("seed", 0) or 0)


# ---------------------------------------------------------------- user presets
PRESET_DIR = APP_DIR / "presets"
STYLE_KEYS = ["cut_speed", "slowmo_amount", "slowmo_freq", "zoom_amount", "effects_amount", "look", "slowmo",
              "slowmo_smoothing", "pulse_mode", "transitions", "order_mode", "fx", "fx_strength", "smart_crop",
              "action_on_beat", "game_filter", "diversity", "min_clip", "max_clip", "beat_mode", "framing",
              "base_zoom"]


def list_user_presets() -> List[str]:
    try:
        return sorted(p.stem for p in PRESET_DIR.glob("*.json"))
    except OSError:
        return []


def save_user_preset(name: str, s: EditSettings) -> None:
    import re
    safe = re.sub(r"[^\w\- ]+", "_", name).strip() or "preset"
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    data = {k: getattr(s, k) for k in STYLE_KEYS}
    (PRESET_DIR / f"{safe}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_user_preset(name: str, s: EditSettings) -> bool:
    try:
        data = json.loads((PRESET_DIR / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for k in STYLE_KEYS:
        if k in data:
            setattr(s, k, list(data[k]) if isinstance(data[k], list) else data[k])
    s.preset = "Custom"
    s.clamp()
    return True
