"""Exception hierarchy. Every error carries a user-friendly message and an optional hint."""
from __future__ import annotations


class AmvError(Exception):
    """Base class for all expected, user-reportable errors."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def user_text(self) -> str:
        return self.message + (f"\n\nTip: {self.hint}" if self.hint else "")


class InputError(AmvError):
    """Bad or missing user input (files, folders, settings)."""


class MediaError(AmvError):
    """A media file could not be read / has no usable stream."""


class FfmpegError(AmvError):
    """ffmpeg is missing or a command failed."""


class AnalysisError(AmvError):
    """Audio / video analysis failed."""


class PlanError(AmvError):
    """The edit could not be planned."""


class RenderError(AmvError):
    """Rendering failed."""


class CancelledError(AmvError):
    """The user cancelled the job."""

    def __init__(self, message: str = "Cancelled by user."):
        super().__init__(message)
