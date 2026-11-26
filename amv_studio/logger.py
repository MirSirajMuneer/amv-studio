"""Logging: rotating file in ~/.amv_studio/logs plus an optional in-memory hook for the GUI."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Callable, Optional

APP_DIR = Path.home() / ".amv_studio"
LOG_DIR = APP_DIR / "logs"
LOG_FILE = LOG_DIR / "amv_studio.log"

_logger: Optional[logging.Logger] = None


class _HookHandler(logging.Handler):
    def __init__(self, hook: Callable[[str, str], None]):
        super().__init__(level=logging.INFO)
        self.hook = hook

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            self.hook(record.levelname, record.getMessage())
        except Exception:
            pass


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    lg = logging.getLogger("amv_studio")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        lg.addHandler(fh)
    except OSError:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        lg.addHandler(sh)
    _logger = lg
    return lg


def attach_hook(hook: Callable[[str, str], None]) -> logging.Handler:
    h = _HookHandler(hook)
    get_logger().addHandler(h)
    return h


def detach_hook(handler: logging.Handler) -> None:
    get_logger().removeHandler(handler)
