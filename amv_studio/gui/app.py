"""AMV Studio GUI (customtkinter).

Smoothness design:
* settings live in TABS (a few dozen widgets visible at a time, no scrolling canvas -> no scroll lag)
* the whole job runs in a separate PROCESS (no shared GIL with the window), results come through a queue
* progress events are batched: the window is updated at most ~15x per second
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

import customtkinter as ctk

try:                                         # optional: drag & drop of files onto the window
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND = True
except Exception:                            # pragma: no cover - depends on the user's install
    DND_FILES, TkinterDnD, _DND = None, None, False

from .. import config
from ..config import (APP_NAME, ASPECTS, AUDIO_EXTS, BEAT_MODES, DURATION_CHOICES, ENCODERS, FPS_CHOICES, FRAMING_LABELS,
                      FRAMINGS, FX, FX_LABELS, LOOKS, ORDER_LABELS, ORDER_MODES, PLATFORMS, PRESETS, PULSE_LABELS,
                      PULSE_MODES, QUALITIES, RESOLUTION_LABELS, RESOLUTIONS, SMOOTHING, SMOOTHING_LABELS,
                      TEXT_POSITIONS, TRANSITION_LABELS, TRANSITIONS, VIDEO_EXTS, EditSettings, load_project,
                      list_user_presets, load_settings, load_user_preset, output_size, parse_size, save_project,
                      save_settings, save_user_preset)
from ..errors import InputError
from ..ffmpeg_utils import fmt_time
from ..logger import APP_DIR, LOG_FILE, get_logger
from ..pipeline import DEFAULT_OUTPUT, STEPS
from ..worker import start_job

log = get_logger()

ACCENT = "#ff3d71"
ACCENT_HOVER = "#e0305f"
CARD = "#1c1d2b"
BG = "#12131c"
FIELD = "#14151f"
BTN = "#2c2e42"
MUTED = "#9aa0b4"
OK = "#3ddc97"
WARN = "#ffb020"
ERR = "#ff5c5c"
USER_PRESET_PREFIX = "★ "


def parse_duration(text: str) -> float:
    """Accepts '45', '45s', '1:30', '2m'."""
    t = text.strip().lower().replace(" ", "")
    if not t:
        raise ValueError("empty")
    if ":" in t:
        m, s = t.split(":", 1)
        return int(m) * 60 + float(s)
    if t.endswith("min"):
        return float(t[:-3]) * 60
    if t.endswith("m"):
        return float(t[:-1]) * 60
    if t.endswith("s"):
        return float(t[:-1])
    return float(t)


class StepRow(ctk.CTkFrame):
    COLORS = {"pending": MUTED, "running": ACCENT, "done": OK, "error": ERR}
    ICONS = {"pending": "○", "running": "◉", "done": "✔", "error": "✖"}

    def __init__(self, master, title: str):
        super().__init__(master, fg_color="transparent")
        self.columnconfigure(1, weight=1)
        self.icon = ctk.CTkLabel(self, text="○", width=26, text_color=MUTED, font=ctk.CTkFont(size=17))
        self.icon.grid(row=0, column=0, rowspan=2, padx=(2, 6))
        self.title = ctk.CTkLabel(self, text=title, anchor="w", font=ctk.CTkFont(size=13, weight="bold"))
        self.title.grid(row=0, column=1, sticky="w")
        self.pct = ctk.CTkLabel(self, text="0%", width=48, anchor="e", text_color=MUTED)
        self.pct.grid(row=0, column=2, padx=(4, 4))
        self.bar = ctk.CTkProgressBar(self, height=9, progress_color=ACCENT)
        self.bar.set(0)
        self.bar.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 4), pady=(2, 0))
        self.msg = ctk.CTkLabel(self, text="", anchor="w", text_color=MUTED, font=ctk.CTkFont(size=11))
        self.msg.grid(row=2, column=1, columnspan=2, sticky="ew")
        self.state = "pending"
        self._pct_text = "0%"
        self._msg_text = ""

    def set_state(self, state: str) -> None:
        if state == self.state and state != "done":
            return
        self.state = state
        self.icon.configure(text=self.ICONS[state], text_color=self.COLORS[state])
        self.bar.configure(progress_color=OK if state == "done" else (ERR if state == "error" else ACCENT))
        if state == "done":
            self.bar.set(1.0)
            self._set_pct("100%")

    def _set_pct(self, txt: str) -> None:
        if txt != self._pct_text:
            self._pct_text = txt
            self.pct.configure(text=txt)

    def set_message(self, msg: str) -> None:
        msg = msg[:90]
        if msg != self._msg_text:
            self._msg_text = msg
            self.msg.configure(text=msg)

    def set_progress(self, frac: float, msg: str = "") -> None:
        frac = min(1.0, max(0.0, frac))
        self.bar.set(frac)
        self._set_pct(f"{int(frac * 100)}%")
        if msg:
            self.set_message(msg)

    def reset(self) -> None:
        self.state = "x"
        self.set_state("pending")
        self.bar.set(0)
        self._set_pct("0%")
        self.set_message("")


_BASES = (ctk.CTk, TkinterDnD.DnDWrapper) if _DND else (ctk.CTk,)


class App(*_BASES):  # type: ignore[misc]
    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.settings = load_settings()
        try:
            ctk.set_widget_scaling(self.settings.ui_scale)
            ctk.set_window_scaling(self.settings.ui_scale)
        except Exception:
            pass
        self.dnd_ok = False
        if _DND:
            try:
                self.TkdndVersion = TkinterDnD._require(self)
                self.dnd_ok = True
            except Exception as e:                 # pragma: no cover
                log.warning("Drag & drop unavailable: %s", e)
        self.title(f"{APP_NAME}  ·  Automatic Beat-Synced AMV Maker")
        self.geometry("1360x880")
        self.minsize(1180, 780)
        self.configure(fg_color=BG)

        self.job = None
        self.running = False
        self.is_preview = False
        self.t_start = 0.0
        self.step_frac: Dict[str, float] = {}
        self.outputs: List[str] = []
        self.last_report = ""
        self.last_seed = 0
        self.preview_info: Optional[tuple] = None     # (fingerprint, seed) of the last preview
        self._lockable: list = []
        self._pending_logs: List[tuple] = []

        self._build_vars()
        self._build_ui()
        self._apply_settings_to_ui(self.settings)
        self._refresh_size_label()
        if self.dnd_ok:
            try:
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self._on_drop)
            except Exception as e:                 # pragma: no cover
                log.warning("Could not enable drag & drop: %s", e)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(60, self._poll)

    # ================================================================= variables
    def _build_vars(self) -> None:
        self.v_out = tk.StringVar()
        self.v_dur: Dict[float, tk.BooleanVar] = {secs: tk.BooleanVar() for _, secs in DURATION_CHOICES}
        self.v_custom = tk.StringVar()
        self.v_variations = tk.StringVar(value="1")
        self.v_preset = tk.StringVar()
        self.v_platform = tk.StringVar(value="None")
        self.v_aspect = tk.StringVar()
        self.v_res = tk.StringVar()
        self.v_size = tk.StringVar()
        self.v_fps = tk.StringVar()
        self.v_framing = tk.StringVar()
        self.v_look = tk.StringVar()
        self.v_quality = tk.StringVar()
        self.v_encoder = tk.StringVar()
        self.v_workers = tk.StringVar()
        self.v_pulse = tk.StringVar()
        self.v_beat = tk.StringVar()
        self.v_order = tk.StringVar()
        self.v_smooth = tk.StringVar()
        self.v_titlepos = tk.StringVar()
        self.v_title = tk.StringVar()
        self.v_titlesecs = tk.StringVar()
        self.v_outro = tk.StringVar()
        self.v_outrosecs = tk.StringVar()
        self.v_mark = tk.StringVar()
        self.v_musicstart = tk.StringVar()
        self.v_cut = tk.DoubleVar()
        self.v_minclip = tk.DoubleVar()
        self.v_maxclip = tk.DoubleVar()
        self.v_slow = tk.DoubleVar()
        self.v_slowfreq = tk.DoubleVar()
        self.v_zoom = tk.DoubleVar()
        self.v_fx = tk.DoubleVar()
        self.v_diversity = tk.DoubleVar()
        self.v_fxstrength = tk.DoubleVar()
        self.v_basezoom = tk.DoubleVar()
        self.v_gameaudio = tk.DoubleVar()
        self.v_nudge = tk.DoubleVar()
        self.v_slowmo = tk.BooleanVar()
        self.v_onbeat = tk.BooleanVar()
        self.v_gta = tk.BooleanVar()
        self.v_smartcrop = tk.BooleanVar()
        self.v_autosong = tk.BooleanVar()
        self.v_norm = tk.BooleanVar()
        self.v_trans: Dict[str, tk.BooleanVar] = {t: tk.BooleanVar() for t in TRANSITIONS}
        self.v_fxlist: Dict[str, tk.BooleanVar] = {f: tk.BooleanVar() for f in FX}
        self.v_status = tk.StringVar(value="Ready. Add your footage and a song, then press GENERATE.")
        self.v_overall = tk.StringVar(value="0%")
        self.v_sizelabel = tk.StringVar(value="")
        self.v_scale = tk.StringVar(value="100%")
        self._slider_labels: List[tuple] = []

    # ================================================================= UI helpers
    def _lock(self, w):
        self._lockable.append(w)
        return w

    def _section(self, parent, text: str, row: int, col: int = 0, span: int = 2) -> None:
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
                     text_color=ACCENT).grid(row=row, column=col, columnspan=span, sticky="w", padx=6, pady=(10, 2))

    def _menu(self, parent, row: int, col: int, label: str, var, values: list, command=None, width: int = 200):
        ctk.CTkLabel(parent, text=label, text_color=MUTED, anchor="w").grid(row=row, column=col, sticky="w",
                                                                           padx=(8, 6), pady=3)
        m = self._lock(ctk.CTkOptionMenu(parent, values=values, variable=var, width=width, fg_color=BTN,
                                         button_color=BTN, button_hover_color=ACCENT_HOVER,
                                         command=command or (lambda _=None: self._custom())))
        m.grid(row=row, column=col + 1, sticky="w", padx=(0, 8), pady=3)
        return m

    def _slider(self, parent, row: int, col: int, title: str, hint: str, var: tk.DoubleVar,
                lo: float = 0.0, hi: float = 1.0, fmt: str = "pct", steps: int = 20) -> None:
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row, column=col, sticky="ew", padx=6, pady=(4, 0))
        box.columnconfigure(0, weight=1)
        ctk.CTkLabel(box, text=title, anchor="w").grid(row=0, column=0, sticky="w")
        val = ctk.CTkLabel(box, text="", width=60, anchor="e", text_color=MUTED)
        val.grid(row=0, column=1, sticky="e")

        def text_for(v: float) -> str:
            if fmt == "pct":
                return f"{int(round(float(v) * 100))}%"
            if fmt == "sec":
                return f"{float(v):.2f} s"
            if fmt == "ms":
                v = int(round(float(v)))
                return f"{v:+d} ms" if v else "0 ms"
            return f"{float(v):.2f}"

        def changed(v: float) -> None:
            t = text_for(v)
            if t != getattr(val, "_t", None):
                val._t = t                           # type: ignore[attr-defined]
                val.configure(text=t)
            self._custom(slider=True)

        s = ctk.CTkSlider(box, from_=lo, to=hi, number_of_steps=steps, variable=var, command=changed,
                          progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER)
        s.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 0))
        if hint:
            ctk.CTkLabel(box, text=hint, text_color=MUTED, font=ctk.CTkFont(size=10), anchor="w").grid(
                row=2, column=0, columnspan=2, sticky="w")
        self._lock(s)
        self._slider_labels.append((var, val, text_for))

    def _switch(self, parent, row: int, col: int, text: str, var, span: int = 2):
        sw = self._lock(ctk.CTkSwitch(parent, text=text, variable=var, progress_color=ACCENT,
                                      command=self._custom))
        sw.grid(row=row, column=col, columnspan=span, sticky="w", padx=8, pady=4)
        return sw

    def _entry(self, parent, var, width: int = 200, placeholder: str = ""):
        return self._lock(ctk.CTkEntry(parent, textvariable=var, width=width, placeholder_text=placeholder))

    def _button(self, parent, text: str, command, width: int = 110, accent: bool = False, **kw):
        return self._lock(ctk.CTkButton(parent, text=text, width=width, command=command,
                                        fg_color=ACCENT if accent else BTN,
                                        hover_color=ACCENT_HOVER if accent else "#3a3d58", **kw))

    # ================================================================= UI
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=3, uniform="c")
        self.columnconfigure(1, weight=2, uniform="c")
        self.rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(12, 4))
        ctk.CTkLabel(header, text="AMV STUDIO", font=ctk.CTkFont(size=26, weight="bold"),
                     text_color=ACCENT).pack(side="left")
        ctk.CTkLabel(header, text=f"v{config.VERSION}  ·  footage + music → beat-synced edits",
                     text_color=MUTED).pack(side="left", padx=14, pady=(8, 0))
        hb = ctk.CTkFrame(header, fg_color="transparent")
        hb.pack(side="right")
        for text, cmd in (("Open project", self._open_project), ("Save project", self._save_project),
                          ("Save preset", self._save_preset), ("Support", self._support_bundle),
                          ("About", self._about)):
            self._lock(ctk.CTkButton(hb, text=text, width=92, height=28, fg_color=BTN, hover_color="#3a3d58",
                                     command=cmd)).pack(side="left", padx=3)
        ctk.CTkLabel(hb, text="UI size", text_color=MUTED).pack(side="left", padx=(12, 4))
        ctk.CTkOptionMenu(hb, values=["90%", "100%", "110%", "125%"], variable=self.v_scale, width=76, height=28,
                          fg_color=BTN, button_color=BTN, command=self._on_ui_scale).pack(side="left")

        # ---- left: tabs
        self.tabs = ctk.CTkTabview(self, fg_color=CARD, corner_radius=14, segmented_button_selected_color=ACCENT,
                                   segmented_button_selected_hover_color=ACCENT_HOVER)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=(14, 6), pady=(0, 14))
        names = ["Media", "Style", "Effects", "Output", "Text & Audio"]
        frames = {n: self.tabs.add(n) for n in names}
        self._build_media(frames["Media"])
        self._build_style(frames["Style"])
        self._build_effects(frames["Effects"])
        self._build_output(frames["Output"])
        self._build_text_audio(frames["Text & Audio"])

        # ---- right: pipeline
        right = ctk.CTkFrame(self, fg_color=CARD, corner_radius=14)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 14), pady=(0, 14))
        self._build_pipeline(right)

    # ---------------------------------------------------------------- Media tab
    def _build_media(self, p) -> None:
        p.columnconfigure(0, weight=1)
        ctk.CTkLabel(p, text="Video files (long gameplay, clips…)" + ("  —  or drop files here" if self.dnd_ok else ""),
                     anchor="w", text_color=MUTED).grid(row=0, column=0, sticky="w", padx=6)
        self.lb = tk.Listbox(p, selectmode="extended", height=5, bg=FIELD, fg="#e8e8f0", selectbackground=ACCENT,
                             selectforeground="white", highlightthickness=0, borderwidth=0, activestyle="none",
                             font=("Segoe UI", 10))
        self.lb.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 4))
        b = ctk.CTkFrame(p, fg_color="transparent")
        b.grid(row=2, column=0, sticky="w", padx=6)
        self._button(b, "＋ Add videos", self._add_videos, 120, accent=True).pack(side="left", padx=(0, 6))
        self._button(b, "Remove selected", lambda: self._remove_sel(self.lb), 120).pack(side="left", padx=(0, 6))
        self._button(b, "Clear", lambda: self.lb.delete(0, "end"), 70).pack(side="left")

        ctk.CTkLabel(p, text="Music (several songs = one edit per song)", anchor="w", text_color=MUTED).grid(
            row=3, column=0, sticky="w", padx=6, pady=(12, 0))
        self.lbm = tk.Listbox(p, selectmode="extended", height=3, bg=FIELD, fg="#e8e8f0", selectbackground=ACCENT,
                              selectforeground="white", highlightthickness=0, borderwidth=0, activestyle="none",
                              font=("Segoe UI", 10))
        self.lbm.grid(row=4, column=0, sticky="ew", padx=6, pady=(2, 4))
        b2 = ctk.CTkFrame(p, fg_color="transparent")
        b2.grid(row=5, column=0, sticky="w", padx=6)
        self._button(b2, "＋ Add songs", self._add_music, 120, accent=True).pack(side="left", padx=(0, 6))
        self._button(b2, "Remove selected", lambda: self._remove_sel(self.lbm), 120).pack(side="left", padx=(0, 6))
        self._button(b2, "Clear", lambda: self.lbm.delete(0, "end"), 70).pack(side="left")

        ctk.CTkLabel(p, text="Save finished videos to", anchor="w", text_color=MUTED).grid(
            row=6, column=0, sticky="w", padx=6, pady=(12, 0))
        r = ctk.CTkFrame(p, fg_color="transparent")
        r.grid(row=7, column=0, sticky="ew", padx=6)
        r.columnconfigure(0, weight=1)
        self._lock(ctk.CTkEntry(r, textvariable=self.v_out)).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._button(r, "Browse", self._pick_out, 90).grid(row=0, column=1)

        ctk.CTkLabel(p, text="Video length (tick several to make them all at once)", anchor="w",
                     text_color=MUTED).grid(row=8, column=0, sticky="w", padx=6, pady=(12, 0))
        g = ctk.CTkFrame(p, fg_color="transparent")
        g.grid(row=9, column=0, sticky="w", padx=2)
        for i, (label, secs) in enumerate(DURATION_CHOICES):
            self._lock(ctk.CTkCheckBox(g, text=label, variable=self.v_dur[secs], width=80, fg_color=ACCENT,
                                       hover_color=ACCENT_HOVER)).grid(row=i // 4, column=i % 4, padx=4, pady=3,
                                                                       sticky="w")
        r2 = ctk.CTkFrame(p, fg_color="transparent")
        r2.grid(row=10, column=0, sticky="w", padx=6, pady=(6, 0))
        ctk.CTkLabel(r2, text="Custom length(s):", text_color=MUTED).pack(side="left")
        self._entry(r2, self.v_custom, 130, "45 or 1:15, 2:30").pack(side="left", padx=8)
        ctk.CTkLabel(r2, text="Variations per length:", text_color=MUTED).pack(side="left", padx=(12, 4))
        self._lock(ctk.CTkOptionMenu(r2, values=["1", "2", "3", "4", "5"], variable=self.v_variations, width=64,
                                     fg_color=BTN, button_color=BTN)).pack(side="left")

    # ---------------------------------------------------------------- Style tab
    def _build_style(self, p) -> None:
        p.columnconfigure(0, weight=1, uniform="s")
        p.columnconfigure(1, weight=1, uniform="s")
        top = ctk.CTkFrame(p, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))
        ctk.CTkLabel(top, text="Style preset", text_color=MUTED).pack(side="left", padx=(8, 6))
        self.preset_menu = self._lock(ctk.CTkOptionMenu(top, values=self._preset_names(), variable=self.v_preset,
                                                        command=self._on_preset, width=280, fg_color=ACCENT,
                                                        button_color=ACCENT_HOVER, button_hover_color=ACCENT_HOVER))
        self.preset_menu.pack(side="left")

        left = ctk.CTkFrame(p, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        self._slider(left, 0, 0, "Cut speed", "calm  →  frantic", self.v_cut)
        self._slider(left, 1, 0, "Shortest shot", "limit for fast cuts", self.v_minclip, 0.2, 3.0, "sec", 28)
        self._slider(left, 2, 0, "Longest shot", "limit for slow parts", self.v_maxclip, 1.0, 12.0, "sec", 22)
        self._slider(left, 3, 0, "Slow-motion depth", "subtle  →  dramatic", self.v_slow)
        self._slider(left, 4, 0, "Slow-motion how often", "rare  →  often", self.v_slowfreq)
        self._slider(left, 5, 0, "Zoom punch amount", "none  →  heavy", self.v_zoom)
        self._slider(left, 6, 0, "Transitions how often", "few  →  many", self.v_fx)
        self._slider(left, 7, 0, "Footage variety", "best only  →  spread out", self.v_diversity)

        right = ctk.CTkFrame(p, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew")
        self._menu(right, 0, 0, "Colour look", self.v_look, LOOKS, width=170)
        self._menu(right, 1, 0, "Beat pulse", self.v_pulse, [PULSE_LABELS[m] for m in PULSE_MODES], width=170)
        self._menu(right, 2, 0, "Beat feel", self.v_beat, BEAT_MODES, width=170)
        self._menu(right, 3, 0, "Clip order", self.v_order, [ORDER_LABELS[m] for m in ORDER_MODES], width=170)
        self._menu(right, 4, 0, "Slow-mo smoothing", self.v_smooth, [SMOOTHING_LABELS[m] for m in SMOOTHING], width=170)
        self._switch(right, 5, 0, "Slow-motion speed ramps", self.v_slowmo)
        self._switch(right, 6, 0, "Action hits exactly on the beat", self.v_onbeat)
        self._switch(right, 7, 0, "Skip menus, maps & idle driving", self.v_gta)
        self._switch(right, 8, 0, "Smart crop (follow the action)", self.v_smartcrop)

    # ---------------------------------------------------------------- Effects tab
    def _build_effects(self, p) -> None:
        self._section(p, "Transitions (the app chooses from the ticked ones)", 0)
        g = ctk.CTkFrame(p, fg_color="transparent")
        g.grid(row=1, column=0, columnspan=2, sticky="w")
        for i, t in enumerate(TRANSITIONS):
            self._lock(ctk.CTkCheckBox(g, text=TRANSITION_LABELS[t], variable=self.v_trans[t], width=150,
                                       fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._custom)
                       ).grid(row=i // 3, column=i % 3, padx=6, pady=4, sticky="w")
        bb = ctk.CTkFrame(p, fg_color="transparent")
        bb.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self._button(bb, "All", lambda: self._set_all(self.v_trans, True), 60).pack(side="left", padx=6)
        self._button(bb, "None", lambda: self._set_all(self.v_trans, False), 60).pack(side="left")

        self._section(p, "Look effects (stack as many as you like, applied to every clip)", 3)
        g2 = ctk.CTkFrame(p, fg_color="transparent")
        g2.grid(row=4, column=0, columnspan=2, sticky="w")
        for i, f in enumerate(FX):
            self._lock(ctk.CTkCheckBox(g2, text=FX_LABELS[f], variable=self.v_fxlist[f], width=170, fg_color=ACCENT,
                                       hover_color=ACCENT_HOVER, command=self._custom)
                       ).grid(row=i // 3, column=i % 3, padx=6, pady=4, sticky="w")
        bb2 = ctk.CTkFrame(p, fg_color="transparent")
        bb2.grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self._button(bb2, "All", lambda: self._set_all(self.v_fxlist, True), 60).pack(side="left", padx=6)
        self._button(bb2, "None", lambda: self._set_all(self.v_fxlist, False), 60).pack(side="left")
        sl = ctk.CTkFrame(p, fg_color="transparent")
        sl.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        sl.columnconfigure(0, weight=1)
        self._slider(sl, 0, 0, "Effect strength", "subtle  →  strong", self.v_fxstrength)
        ctk.CTkLabel(p, text="Tip: for realistic gameplay keep it light — Vignette + Extra sharpen + a little Glow "
                             "looks most professional. Heavy effects suit phonk-style edits.",
                     text_color=MUTED, wraplength=640, justify="left", anchor="w").grid(
            row=7, column=0, columnspan=2, sticky="w", padx=8, pady=(10, 0))

    # ---------------------------------------------------------------- Output tab
    def _build_output(self, p) -> None:
        p.columnconfigure(1, weight=1)
        self._menu(p, 0, 0, "Platform preset", self.v_platform, list(PLATFORMS.keys()),
                   command=self._on_platform, width=220)
        self._menu(p, 1, 0, "Format", self.v_aspect, ASPECTS, command=lambda _=None: self._size_changed(), width=220)
        self._menu(p, 2, 0, "Resolution (scale)", self.v_res, [RESOLUTION_LABELS[r] for r in RESOLUTIONS],
                   command=lambda _=None: self._size_changed(), width=220)
        ctk.CTkLabel(p, text="Custom size  W × H", text_color=MUTED, anchor="w").grid(row=3, column=0, sticky="w",
                                                                                  padx=(8, 6), pady=3)
        ce = self._entry(p, self.v_size, 220, "e.g. 1440x2560 (optional)")
        ce.grid(row=3, column=1, sticky="w", pady=3)
        try:
            self.v_size.trace_add("write", lambda *_: self._refresh_size_label())
        except Exception:
            pass
        self._menu(p, 4, 0, "Frame rate", self.v_fps, [str(f) for f in FPS_CHOICES],
                   command=lambda _=None: self._custom(), width=220)
        self._menu(p, 5, 0, "Framing", self.v_framing, [FRAMING_LABELS[f] for f in FRAMINGS],
                   command=lambda _=None: self._custom(), width=220)
        sc = ctk.CTkFrame(p, fg_color="transparent")
        sc.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        sc.columnconfigure(0, weight=1)
        self._slider(sc, 0, 0, "Footage scale (zoom into the picture)", "100 % = full frame  →  140 %",
                     self.v_basezoom, 1.0, 1.4, "pct", 40)
        ctk.CTkLabel(p, textvariable=self.v_sizelabel, text_color=OK, anchor="w",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(row=7, column=0, columnspan=2, sticky="w",
                                                                    padx=8, pady=(8, 2))
        self._section(p, "Speed & quality", 8)
        self._menu(p, 9, 0, "Render quality", self.v_quality, QUALITIES, command=lambda _=None: None, width=220)
        self._menu(p, 10, 0, "Encoder", self.v_encoder, ENCODERS, command=lambda _=None: None, width=220)
        self._menu(p, 11, 0, "Parallel clips", self.v_workers, ["Auto", "1", "2", "3", "4", "6"],
                   command=lambda _=None: None, width=220)
        ctk.CTkLabel(p, text="Balanced is best for final videos. Fast is for quick tests. 4K/2K need a fast PC "
                             "and lots of disk space.", text_color=MUTED, wraplength=620, justify="left",
                     anchor="w").grid(row=12, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 0))

    # ---------------------------------------------------------------- Text & Audio tab
    def _build_text_audio(self, p) -> None:
        p.columnconfigure(1, weight=1)
        self._section(p, "Text on the video (optional)", 0)
        ctk.CTkLabel(p, text="Title", text_color=MUTED, anchor="w").grid(row=1, column=0, sticky="w", padx=8, pady=3)
        r = ctk.CTkFrame(p, fg_color="transparent")
        r.grid(row=1, column=1, sticky="w")
        self._entry(r, self.v_title, 250, "e.g. NOV 19 • JASON + LUCIA").pack(side="left", padx=(0, 6))
        self._lock(ctk.CTkOptionMenu(r, values=TEXT_POSITIONS, variable=self.v_titlepos, width=90, fg_color=BTN,
                                     button_color=BTN)).pack(side="left", padx=(0, 6))
        self._entry(r, self.v_titlesecs, 46).pack(side="left")
        ctk.CTkLabel(r, text="sec", text_color=MUTED).pack(side="left", padx=4)
        ctk.CTkLabel(p, text="End text", text_color=MUTED, anchor="w").grid(row=2, column=0, sticky="w", padx=8, pady=3)
        r2 = ctk.CTkFrame(p, fg_color="transparent")
        r2.grid(row=2, column=1, sticky="w")
        self._entry(r2, self.v_outro, 250, "e.g. SUBSCRIBE FOR MORE").pack(side="left", padx=(0, 6))
        self._entry(r2, self.v_outrosecs, 46).pack(side="left")
        ctk.CTkLabel(r2, text="sec", text_color=MUTED).pack(side="left", padx=4)
        ctk.CTkLabel(p, text="Watermark", text_color=MUTED, anchor="w").grid(row=3, column=0, sticky="w", padx=8, pady=3)
        self._entry(p, self.v_mark, 250, "@yourchannel").grid(row=3, column=1, sticky="w", pady=3)

        self._section(p, "Music & sound", 4)
        sl = ctk.CTkFrame(p, fg_color="transparent")
        sl.grid(row=5, column=0, columnspan=2, sticky="ew")
        sl.columnconfigure(0, weight=1)
        self._slider(sl, 0, 0, "Original game sound under the music", "off  →  loud", self.v_gameaudio)
        self._slider(sl, 1, 0, "Sync fine-tune", "cuts earlier  ←  0  →  cuts later", self.v_nudge, -100, 100, "ms", 40)
        self._switch(p, 6, 0, "Auto-pick the best part of the song", self.v_autosong)
        ms = ctk.CTkFrame(p, fg_color="transparent")
        ms.grid(row=7, column=0, columnspan=2, sticky="w")
        ctk.CTkLabel(ms, text="  or start the song at (seconds):", text_color=MUTED).pack(side="left", padx=8)
        self._entry(ms, self.v_musicstart, 70).pack(side="left")
        self._switch(p, 8, 0, "Normalize loudness for social media", self.v_norm)

    # ---------------------------------------------------------------- Progress panel
    def _build_pipeline(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        ctk.CTkLabel(parent, text="Progress", font=ctk.CTkFont(size=15, weight="bold"), anchor="w"
                     ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        steps = ctk.CTkFrame(parent, fg_color="transparent")
        steps.grid(row=1, column=0, sticky="ew", padx=12)
        steps.columnconfigure(0, weight=1)
        self.rows: Dict[str, StepRow] = {}
        for i, (sid, label, _w) in enumerate(STEPS):
            r = StepRow(steps, f"{i + 1}. {label}")
            r.grid(row=i, column=0, sticky="ew", pady=3)
            self.rows[sid] = r

        overall = ctk.CTkFrame(parent, fg_color="transparent")
        overall.grid(row=2, column=0, sticky="ew", padx=16, pady=(10, 4))
        overall.columnconfigure(0, weight=1)
        top = ctk.CTkFrame(overall, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="Overall", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(top, textvariable=self.v_overall, text_color=MUTED, anchor="e").grid(row=0, column=1, sticky="e")
        self.overall_bar = ctk.CTkProgressBar(overall, height=14, progress_color=ACCENT)
        self.overall_bar.set(0)
        self.overall_bar.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        self.log_box = ctk.CTkTextbox(parent, fg_color="#10111a", text_color="#c9cde0", wrap="word",
                                      font=ctk.CTkFont(family="Consolas", size=11))
        self.log_box.grid(row=3, column=0, sticky="nsew", padx=14, pady=(6, 6))
        self.log_box.configure(state="disabled")
        try:
            tb = self.log_box._textbox  # noqa: SLF001 - colour tags need the raw widget
            tb.tag_config("ERROR", foreground=ERR)
            tb.tag_config("WARNING", foreground=WARN)
            tb.tag_config("OK", foreground=OK)
        except Exception:
            pass

        ctk.CTkLabel(parent, textvariable=self.v_status, anchor="w", wraplength=500, justify="left",
                     text_color=MUTED).grid(row=4, column=0, sticky="ew", padx=16)
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=5, column=0, sticky="ew", padx=14, pady=(8, 6))
        bar.columnconfigure(0, weight=1)
        self.btn_go = ctk.CTkButton(bar, text="⚡  GENERATE", height=46, font=ctk.CTkFont(size=16, weight="bold"),
                                    fg_color=ACCENT, hover_color=ACCENT_HOVER, command=lambda: self._start(False))
        self.btn_go.grid(row=0, column=0, sticky="ew")
        self.btn_prev = ctk.CTkButton(bar, text="👁 Preview", width=100, height=46, fg_color="#3a3c54",
                                      command=lambda: self._start(True))
        self.btn_prev.grid(row=0, column=1, padx=(8, 0))
        self.btn_cancel = ctk.CTkButton(bar, text="Cancel", width=80, height=46, fg_color="#3a3c54", state="disabled",
                                        command=self._cancel)
        self.btn_cancel.grid(row=0, column=2, padx=(8, 0))
        bar2 = ctk.CTkFrame(parent, fg_color="transparent")
        bar2.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 14))
        for i, (t, c) in enumerate((("Open folder", self._open_folder), ("Edit report", self._show_report),
                                    ("Log file", self._open_log))):
            bar2.columnconfigure(i, weight=1)
            ctk.CTkButton(bar2, text=t, height=32, fg_color=BTN, hover_color="#3a3d58", command=c).grid(
                row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))

    # ================================================================= settings <-> UI
    def _preset_names(self) -> List[str]:
        return list(PRESETS.keys()) + [USER_PRESET_PREFIX + n for n in list_user_presets()]

    def _custom(self, slider: bool = False) -> None:
        if self.v_preset.get() != "Custom" and not self.v_preset.get().startswith(USER_PRESET_PREFIX):
            self.v_preset.set("Custom")
        elif self.v_preset.get().startswith(USER_PRESET_PREFIX):
            self.v_preset.set("Custom")

    def _refresh_slider_labels(self) -> None:
        for var, lab, fn in self._slider_labels:
            try:
                t = fn(var.get())
                lab._t = t                      # type: ignore[attr-defined]
                lab.configure(text=t)
            except Exception:
                pass

    def _set_all(self, d: Dict[str, tk.BooleanVar], value: bool) -> None:
        for v in d.values():
            v.set(value)
        self._custom()

    def _on_preset(self, name: str) -> None:
        s = self._collect(validate=False)
        if name.startswith(USER_PRESET_PREFIX):
            if not load_user_preset(name[len(USER_PRESET_PREFIX):], s):
                messagebox.showwarning("Preset", "That preset could not be loaded.")
                return
        else:
            s.apply_preset(name)
        self._apply_settings_to_ui(s)
        self.v_preset.set(name)

    def _on_platform(self, name: str) -> None:
        s = self._collect(validate=False)
        s.apply_platform(name)
        s.custom_size = ""
        keep_preset = self.v_preset.get()
        self._apply_settings_to_ui(s)
        self.v_preset.set(keep_preset)
        self.v_platform.set(name)
        self._refresh_size_label()

    def _size_changed(self) -> None:
        self.v_platform.set("None") if self.v_platform.get() != "None" else None
        self._refresh_size_label()

    def _refresh_size_label(self) -> None:
        try:
            cs = self.v_size.get().strip()
            if cs and parse_size(cs) is None:
                self.v_sizelabel.set("⚠ Custom size must look like 1440x2560 — it will be ignored")
                return
            s = self._collect(validate=False)
            w, h = output_size(s)
            self.v_sizelabel.set(f"Output: {w} × {h} px   ·   {s.fps} fps   ·   {w * h / 1e6:.1f} megapixels")
        except Exception:
            pass

    def _on_ui_scale(self, value: str) -> None:
        try:
            f = float(value.strip("%")) / 100.0
            ctk.set_widget_scaling(f)
            ctk.set_window_scaling(f)
            self.settings.ui_scale = f
        except Exception:
            pass

    def _apply_settings_to_ui(self, s: EditSettings) -> None:
        self.v_out.set(s.output_dir or str(DEFAULT_OUTPUT))
        for secs, var in self.v_dur.items():
            var.set(secs in s.durations)
        extra = [d for d in s.durations if d not in self.v_dur]
        self.v_custom.set(", ".join(f"{d:g}" for d in extra))
        self.v_variations.set(str(s.variations))
        self.v_preset.set(s.preset if s.preset in PRESETS else "Custom")
        self.v_platform.set(s.platform)
        self.v_aspect.set(s.aspect)
        self.v_res.set(RESOLUTION_LABELS[s.resolution])
        self.v_size.set(s.custom_size)
        self.v_fps.set(str(s.fps))
        self.v_framing.set(FRAMING_LABELS[s.framing])
        self.v_look.set(s.look)
        self.v_quality.set(s.quality)
        self.v_encoder.set(s.encoder)
        self.v_workers.set("Auto" if not s.workers else str(s.workers))
        self.v_pulse.set(PULSE_LABELS[s.pulse_mode])
        self.v_beat.set(s.beat_mode)
        self.v_order.set(ORDER_LABELS[s.order_mode])
        self.v_smooth.set(SMOOTHING_LABELS[s.slowmo_smoothing])
        self.v_cut.set(s.cut_speed)
        self.v_minclip.set(s.min_clip)
        self.v_maxclip.set(s.max_clip)
        self.v_slow.set(s.slowmo_amount)
        self.v_slowfreq.set(s.slowmo_freq)
        self.v_zoom.set(s.zoom_amount)
        self.v_fx.set(s.effects_amount)
        self.v_diversity.set(s.diversity)
        self.v_fxstrength.set(s.fx_strength)
        self.v_basezoom.set(s.base_zoom)
        self.v_gameaudio.set(s.game_audio)
        self.v_nudge.set(s.sync_nudge_ms)
        self.v_slowmo.set(s.slowmo)
        self.v_onbeat.set(s.action_on_beat)
        self.v_gta.set(s.game_filter)
        self.v_smartcrop.set(s.smart_crop)
        self.v_autosong.set(s.auto_music_section)
        self.v_norm.set(s.normalize_audio)
        self.v_musicstart.set(f"{s.music_start:g}" if s.music_start else "")
        self.v_title.set(s.title_text)
        self.v_titlepos.set(s.title_pos)
        self.v_titlesecs.set(f"{s.title_secs:g}")
        self.v_outro.set(s.outro_text)
        self.v_outrosecs.set(f"{s.outro_secs:g}")
        self.v_mark.set(s.watermark_text)
        for t, var in self.v_trans.items():
            var.set(t in s.transitions)
        for f, var in self.v_fxlist.items():
            var.set(f in s.fx)
        self._refresh_slider_labels()
        self._refresh_size_label()

    @staticmethod
    def _inverse(labels: dict, text: str, default: str) -> str:
        for k, v in labels.items():
            if v == text:
                return k
        return default if text not in labels else text

    @staticmethod
    def _num(text: str, default: float) -> float:
        try:
            return float(str(text).replace(",", ".").strip())
        except ValueError:
            return default

    def _collect(self, validate: bool = True) -> EditSettings:
        durs = [secs for secs, var in self.v_dur.items() if var.get()]
        for part in self.v_custom.get().replace(";", ",").split(","):
            if part.strip():
                try:
                    d = parse_duration(part)
                except ValueError:
                    if validate:
                        raise InputError(f"'{part.strip()}' is not a valid length.",
                                         "Use seconds (45) or minutes:seconds (1:15).")
                    continue
                if validate and not (5 <= d <= 600):
                    raise InputError(f"Length {part.strip()} is out of range.", "Choose between 5 seconds and 10 minutes.")
                durs.append(d)
        if validate and not durs:
            raise InputError("Please choose at least one video length.")
        size_txt = self.v_size.get().strip()
        if validate and size_txt and parse_size(size_txt) is None:
            raise InputError(f"'{size_txt}' is not a valid custom size.", "Use the form 1440x2560, or leave it empty.")
        pm = self._inverse({k: v for k, v in PULSE_LABELS.items()}, self.v_pulse.get(), "sparse")
        order = self._inverse(ORDER_LABELS, self.v_order.get(), "energy")
        smooth = self._inverse(SMOOTHING_LABELS, self.v_smooth.get(), "Blend")
        framing = self._inverse(FRAMING_LABELS, self.v_framing.get(), "fill")
        res = self._inverse(RESOLUTION_LABELS, self.v_res.get(), "1080p")
        wk = self.v_workers.get()
        s = EditSettings(
            durations=durs or [30.0], variations=int(self.v_variations.get() or 1),
            aspect=self.v_aspect.get(), resolution=res, custom_size=size_txt, fps=int(self.v_fps.get() or 30),
            framing=framing, base_zoom=self.v_basezoom.get(), look=self.v_look.get(), platform=self.v_platform.get(),
            preset=self.v_preset.get() if self.v_preset.get() in PRESETS else "Custom",
            cut_speed=self.v_cut.get(), slowmo_amount=self.v_slow.get(), slowmo_freq=self.v_slowfreq.get(),
            zoom_amount=self.v_zoom.get(), effects_amount=self.v_fx.get(), slowmo=self.v_slowmo.get(),
            slowmo_smoothing=smooth, pulse_mode=pm, transitions=[t for t, v in self.v_trans.items() if v.get()],
            smart_crop=self.v_smartcrop.get(), order_mode=order, action_on_beat=self.v_onbeat.get(),
            game_filter=self.v_gta.get(), diversity=self.v_diversity.get(), min_clip=self.v_minclip.get(),
            max_clip=self.v_maxclip.get(), fx=[f for f, v in self.v_fxlist.items() if v.get()],
            fx_strength=self.v_fxstrength.get(), auto_music_section=self.v_autosong.get(),
            music_start=self._num(self.v_musicstart.get(), 0.0), normalize_audio=self.v_norm.get(),
            beat_mode=self.v_beat.get(), sync_nudge_ms=int(round(self.v_nudge.get())),
            game_audio=self.v_gameaudio.get(), title_text=self.v_title.get(), title_pos=self.v_titlepos.get(),
            title_secs=self._num(self.v_titlesecs.get(), 2.5), outro_text=self.v_outro.get(),
            outro_secs=self._num(self.v_outrosecs.get(), 2.0), watermark_text=self.v_mark.get(),
            quality=self.v_quality.get(), encoder=self.v_encoder.get(),
            workers=0 if wk == "Auto" else int(wk), output_dir=self.v_out.get().strip(),
            seed=0, analysis_fps=self.settings.analysis_fps, use_cache=True, ui_scale=self.settings.ui_scale)
        return s.clamp()

    # ================================================================= file pickers / lists
    def _videos(self) -> List[str]:
        return list(self.lb.get(0, "end"))

    def _musics(self) -> List[str]:
        return list(self.lbm.get(0, "end"))

    def _add_unique(self, lb, files) -> None:
        existing = set(lb.get(0, "end"))
        for f in files:
            if f and f not in existing:
                lb.insert("end", f)
                existing.add(f)

    def _add_videos(self) -> None:
        exts = " ".join(f"*{e}" for e in VIDEO_EXTS)
        self._add_unique(self.lb, filedialog.askopenfilenames(
            title="Choose video files", filetypes=[("Video files", exts), ("All files", "*.*")]))

    def _add_music(self) -> None:
        exts = " ".join(f"*{e}" for e in AUDIO_EXTS)
        self._add_unique(self.lbm, filedialog.askopenfilenames(
            title="Choose music", filetypes=[("Audio files", exts), ("All files", "*.*")]))

    @staticmethod
    def _remove_sel(lb) -> None:
        for i in reversed(lb.curselection()):
            lb.delete(i)

    def _on_drop(self, event) -> None:
        try:
            files = list(self.tk.splitlist(event.data))
        except Exception:
            files = [event.data]
        vids = [f for f in files if Path(f).suffix.lower() in VIDEO_EXTS]
        auds = [f for f in files if Path(f).suffix.lower() in AUDIO_EXTS]
        self._add_unique(self.lb, vids)
        self._add_unique(self.lbm, auds)
        self.v_status.set(f"Added {len(vids)} video(s) and {len(auds)} song(s).")

    def _pick_out(self) -> None:
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.v_out.set(d)

    def _open_folder(self) -> None:
        folder = Path(self.outputs[-1]).parent if self.outputs else Path(self.v_out.get() or DEFAULT_OUTPUT)
        folder.mkdir(parents=True, exist_ok=True)
        self._open_path(folder)

    def _open_log(self) -> None:
        self._open_path(LOG_FILE if LOG_FILE.exists() else LOG_FILE.parent)

    @staticmethod
    def _open_path(p: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showinfo("Open", f"Location:\n{p}\n\n({e})")

    # ================================================================= projects, presets, support
    def _save_project(self) -> None:
        f = filedialog.asksaveasfilename(title="Save project", defaultextension=".amvproj",
                                         filetypes=[("AMV Studio project", "*.amvproj")])
        if not f:
            return
        try:
            save_project(f, self._collect(validate=False), self._videos(), self._musics(), self.last_seed)
            self.v_status.set(f"Project saved: {Path(f).name}")
        except Exception as e:
            messagebox.showerror("Save project", f"Could not save the project:\n{e}")

    def _open_project(self) -> None:
        f = filedialog.askopenfilename(title="Open project", filetypes=[("AMV Studio project", "*.amvproj"),
                                                                       ("All files", "*.*")])
        if not f:
            return
        try:
            s, videos, musics, seed = load_project(f)
        except Exception as e:
            messagebox.showerror("Open project", f"This project could not be opened:\n{e}")
            return
        self._apply_settings_to_ui(s)
        self.lb.delete(0, "end")
        self.lbm.delete(0, "end")
        missing = [p for p in videos + musics if not Path(p).exists()]
        self._add_unique(self.lb, videos)
        self._add_unique(self.lbm, musics)
        self.last_seed = seed
        self.v_status.set(f"Project loaded: {Path(f).name}" + (f"  ({len(missing)} file(s) not found)" if missing else ""))

    def _save_preset(self) -> None:
        try:
            name = ctk.CTkInputDialog(text="Name for this style preset:", title="Save preset").get_input()
        except Exception:
            name = None
        if not name or not name.strip():
            return
        try:
            save_user_preset(name.strip(), self._collect(validate=False))
            self.preset_menu.configure(values=self._preset_names())
            self.v_preset.set(USER_PRESET_PREFIX + name.strip())
            self.v_status.set(f"Preset saved: {name.strip()}")
        except Exception as e:
            messagebox.showerror("Save preset", f"Could not save the preset:\n{e}")

    def _support_bundle(self) -> None:
        """Zip logs + settings so a problem can be sent to support."""
        try:
            dest = Path.home() / "Desktop"
            dest = dest if dest.exists() else Path.home()
            z = dest / f"AMV_Studio_support_{time.strftime('%Y%m%d_%H%M%S')}.zip"
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in list((APP_DIR / "logs").glob("*")) + [config.SETTINGS_FILE]:
                    if p.exists():
                        zf.write(p, p.name)
                zf.writestr("info.txt", f"{APP_NAME} {config.VERSION}\nPython {sys.version}\nPlatform {sys.platform}\n")
            messagebox.showinfo("Support bundle", f"Saved:\n{z}\n\nSend this file to support.")
        except Exception as e:
            messagebox.showerror("Support bundle", f"Could not create the bundle:\n{e}")

    def _about(self) -> None:
        messagebox.showinfo(
            f"About {APP_NAME}",
            f"{APP_NAME} {config.VERSION}\nAutomatic beat-synced AMV / short-video maker.\n\n"
            "You are responsible for having the rights to the footage and music you edit.\n"
            f"Logs: {LOG_FILE}")

    def _show_report(self) -> None:
        if not self.last_report:
            messagebox.showinfo("Edit report", "Make a video first - the report explains every cut.")
            return
        try:
            win = ctk.CTkToplevel(self)
            win.title("Edit report - what was cut, where, and why")
            win.geometry("980x560")
            box = ctk.CTkTextbox(win, font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
            box.pack(fill="both", expand=True, padx=10, pady=10)
            box.insert("1.0", self.last_report)
            box.configure(state="disabled")
        except Exception as e:
            messagebox.showinfo("Edit report", f"{self.last_report[:3000]}\n\n({e})")

    # ================================================================= running
    def _set_running(self, running: bool) -> None:
        self.running = running
        state = "disabled" if running else "normal"
        for w in self._lockable:
            try:
                w.configure(state=state)
            except Exception:
                pass
        for lb in (self.lb, self.lbm):
            lb.configure(state=state)
        self.btn_go.configure(state=state, text="Working…" if running else "⚡  GENERATE")
        self.btn_prev.configure(state=state)
        self.btn_cancel.configure(state="normal" if running else "disabled")

    def _start(self, preview: bool = False) -> None:
        if self.running:
            return
        videos, musics = self._videos(), self._musics()
        try:
            if not videos:
                raise InputError("Please add at least one video file.", "Click “＋ Add videos” on the Media tab.")
            if not musics:
                raise InputError("Please add a music file.", "Click “＋ Add songs” on the Media tab.")
            missing = [v for v in videos + musics if not Path(v).is_file()]
            if missing:
                raise InputError(f"File not found:\n{missing[0]}")
            settings = self._collect(validate=True)
            if not settings.transitions:
                log.info("No transitions ticked - the edit will use hard cuts only.")
        except InputError as e:
            messagebox.showwarning("Check your settings", e.user_text())
            return
        # a final render re-uses the seed of the last preview of the SAME edit
        fp = settings.edit_fingerprint() + "|" + "|".join(videos + musics)
        if not preview and self.preview_info and self.preview_info[0] == fp:
            settings.seed = self.preview_info[1]
        self.settings = settings
        save_settings(settings)
        for r in self.rows.values():
            r.reset()
        self.step_frac = {sid: 0.0 for sid, _, _ in STEPS}
        self.outputs = []
        self.is_preview = preview
        self._pending_fp = fp
        self.overall_bar.set(0)
        self.v_overall.set("0%")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.v_status.set("Making a quick low-resolution preview…" if preview else
                          "Working… you can leave this window open and grab a coffee ☕")
        self.t_start = time.time()
        self._set_running(True)
        self._finished = False
        try:
            self.job = start_job(settings, videos, musics, preview)
        except Exception as e:
            self._set_running(False)
            messagebox.showerror("Could not start", f"The job could not be started:\n{e}")

    def _cancel(self) -> None:
        if self.job is not None and self.running:
            self.job.cancel()
            self.btn_cancel.configure(state="disabled")
            self.v_status.set("Cancelling… stopping ffmpeg and cleaning up.")

    # ================================================================= event loop
    def _flush_logs(self) -> None:
        if not self._pending_logs:
            return
        self.log_box.configure(state="normal")
        tb = getattr(self.log_box, "_textbox", None)
        for text, tag in self._pending_logs[-200:]:
            try:
                tb.insert("end", text + "\n", tag)
            except Exception:
                self.log_box.insert("end", text + "\n")
        self._pending_logs = []
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _update_overall(self) -> None:
        total = sum(w * min(1.0, self.step_frac.get(sid, 0.0)) for sid, _, w in STEPS)
        f = total / sum(w for _, _, w in STEPS)
        self.overall_bar.set(f)
        elapsed = time.time() - self.t_start
        txt = f"{int(f * 100)}%  ·  elapsed {fmt_time(elapsed)}"
        if self.running and f > 0.06:
            txt += f"  ·  about {fmt_time(elapsed * (1 - f) / f)} left"
        self.v_overall.set(txt)

    def _poll(self) -> None:
        try:
            if self.job is not None:
                events = self.job.poll()
                if events:
                    latest: Dict[str, dict] = {}
                    for ev in events:                         # batch: only the newest progress per step
                        if ev.get("kind") == "step_progress":
                            latest[ev["step"]] = ev
                        else:
                            for st, e2 in list(latest.items()):
                                self._handle(e2)
                            latest.clear()
                            self._handle(ev)
                    for e2 in latest.values():
                        self._handle(e2)
                    self._update_overall()
                    self._flush_logs()
                elif self.running and not self.job.alive() and not getattr(self, "_finished", False):
                    # the worker process vanished without saying goodbye (crash / killed)
                    self._handle({"kind": "error", "message": "The video engine stopped unexpectedly.\n\n"
                                  "Use the Support button to save the log file."})
                    self._handle({"kind": "finished"})
                if self.running:
                    self._update_overall()
        except Exception:
            log.exception("GUI update error")
        self.after(70, self._poll)

    def _handle(self, ev: dict) -> None:
        k = ev.get("kind")
        step = ev.get("step")
        row = self.rows.get(step) if step else None
        if k == "step_start" and row:
            row.set_state("running")
            if ev.get("message"):
                row.set_message(ev["message"])
        elif k == "step_progress" and row:
            if row.state == "pending":
                row.set_state("running")
            row.set_progress(ev["frac"], ev.get("message", ""))
            self.step_frac[step] = ev["frac"]
        elif k == "step_done" and row:
            row.set_state("done")
            if ev.get("message"):
                row.set_message(ev["message"])
            self.step_frac[step] = 1.0
            self._pending_logs.append((f"✔ {ev.get('message', step)}", "OK"))
        elif k == "log":
            lvl = ev.get("level", "INFO")
            self._pending_logs.append((ev.get("message", ""), lvl if lvl in ("ERROR", "WARNING") else ""))
        elif k == "seed":
            self.last_seed = int(ev.get("seed", 0))
        elif k == "output":
            self.outputs.append(ev["path"])
            self._pending_logs.append((f"★ {ev.get('message', '')}", "OK"))
        elif k == "report":
            self.last_report = ev.get("text", "")
        elif k == "done":
            self.overall_bar.set(1.0)
            self.v_overall.set(f"100%  ·  {ev.get('message', 'Done')}")
            outs = ev.get("outputs", [])
            if ev.get("preview"):
                self.preview_info = (getattr(self, "_pending_fp", ""), int(ev.get("seed", self.last_seed)))
                self.v_status.set("✔ Preview ready. Like it? Press GENERATE to render the SAME edit in full quality.")
                if outs:
                    self._open_path(Path(outs[0]))
            else:
                names = "\n".join(Path(p).name for p in outs)
                self.v_status.set(f"✔ {ev.get('message', 'Done')} — {len(outs)} video(s) saved.")
                self._flush_logs()
                messagebox.showinfo("All done!", f"Your AMV is ready:\n\n{names}")
        elif k == "cancelled":
            self.v_status.set("Cancelled. Temporary files were removed.")
            for r in self.rows.values():
                if r.state == "running":
                    r.set_state("pending")
        elif k == "error":
            msg = ev.get("message", "Unknown error")
            self.v_status.set("✖ Something went wrong — see message.")
            for r in self.rows.values():
                if r.state == "running":
                    r.set_state("error")
            self._pending_logs.append(("✖ " + msg.replace("\n", " "), "ERROR"))
            self._flush_logs()
            messagebox.showerror("Could not finish", msg)
        elif k == "finished":
            self._finished = True
            self._set_running(False)
            self._flush_logs()
            self.job = None

    # ================================================================= close
    def _on_close(self) -> None:
        if self.running:
            if not messagebox.askyesno("Quit?", "A video is still being created. Cancel it and quit?"):
                return
            if self.job is not None:
                self.job.cancel()
                time.sleep(0.6)
                self.job.terminate()
        try:
            s = self._collect(validate=False)
            s.ui_scale = self.settings.ui_scale
            save_settings(s)
            self.settings = s
        except Exception:
            pass
        self.destroy()


def run_gui() -> None:
    app = App()
    app.mainloop()
