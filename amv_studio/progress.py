"""Cancellation token and progress-event helpers shared by every layer."""
from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

from .errors import CancelledError

# A progress callback receives plain dicts so the GUI can consume them from a queue:
#   {"kind": "step_start",    "step": "music", "message": "..."}
#   {"kind": "step_progress", "step": "music", "frac": 0.4, "message": "..."}
#   {"kind": "step_done",     "step": "music", "message": "..."}
#   {"kind": "log",           "level": "INFO", "message": "..."}
#   {"kind": "done",          "outputs": [...]}
#   {"kind": "error",         "message": "...", "fatal": True}
EventCallback = Callable[[Dict], None]


class CancelToken:
    """Thread-safe cancel flag. A child token is also 'set' whenever its parent is."""

    def __init__(self, parent: "Optional[CancelToken]" = None, event=None) -> None:
        # `event` may be a threading.Event or a multiprocessing Event (anything with set / is_set)
        self._ev = event if event is not None else threading.Event()
        self._parent = parent

    def cancel(self) -> None:
        self._ev.set()

    def is_set(self) -> bool:
        return self._ev.is_set() or (self._parent is not None and self._parent.is_set())

    def check(self) -> None:
        if self._ev.is_set():
            raise CancelledError()


class StepReporter:
    """Reports progress for one pipeline step, optionally mapped into a sub-range."""

    def __init__(self, callback: Optional[EventCallback], step: str,
                 lo: float = 0.0, hi: float = 1.0):
        self.callback = callback
        self.step = step
        self.lo = lo
        self.hi = hi
        self._last = -1.0

    def sub(self, lo: float, hi: float) -> "StepReporter":
        span = self.hi - self.lo
        return StepReporter(self.callback, self.step, self.lo + span * lo, self.lo + span * hi)

    def __call__(self, frac: float, message: str = "") -> None:
        if self.callback is None:
            return
        frac = min(1.0, max(0.0, float(frac)))
        g = self.lo + (self.hi - self.lo) * frac
        # throttle: only emit if moved >= 0.5 % or message present at ends
        if message == "" and abs(g - self._last) < 0.005 and frac < 1.0:
            return
        self._last = g
        self.callback({"kind": "step_progress", "step": self.step, "frac": g, "message": message})
