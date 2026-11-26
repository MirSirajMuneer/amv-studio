"""Runs a Pipeline job in a separate PROCESS so the GUI never stutters (no shared Python GIL).

The GUI sends plain data in, and receives progress events through a multiprocessing queue.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
from typing import List, Optional

from .config import EditSettings
from .progress import CancelToken


def _entry(settings_dict: dict, videos: List[str], musics: List[str], preview: bool, q, cancel_event) -> None:
    from .pipeline import Pipeline           # imported in the child process
    try:
        s = EditSettings.from_dict(settings_dict)
        tok = CancelToken(event=cancel_event)
        Pipeline(s, videos, musics, callback=q.put, cancel=tok, preview=preview).run()
    except BaseException:                      # the pipeline already reported errors / cancellation
        pass
    finally:
        q.put({"kind": "finished"})


class JobHandle:
    """Handle for a running job: poll events, cancel, check liveness."""

    def __init__(self, settings: EditSettings, videos: List[str], musics: List[str], preview: bool = False):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue()
        self.cancel_event = ctx.Event()
        self.proc = ctx.Process(target=_entry, daemon=True, name="amv-job",
                                args=(settings.__dict__ | {}, videos, musics, preview, self.q, self.cancel_event))
        self.proc.start()

    def poll(self, limit: int = 300) -> List[dict]:
        out: List[dict] = []
        for _ in range(limit):
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out

    def cancel(self) -> None:
        self.cancel_event.set()

    def alive(self) -> bool:
        return self.proc.is_alive()

    def terminate(self) -> None:
        try:
            if self.proc.is_alive():
                self.proc.terminate()
        except Exception:
            pass


class ThreadJob:
    """Same interface as JobHandle but runs in a thread (fallback if a process cannot be started)."""

    def __init__(self, settings: EditSettings, videos: List[str], musics: List[str], preview: bool = False):
        import threading
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.token = CancelToken()

        def run() -> None:
            from .pipeline import Pipeline
            try:
                Pipeline(settings, videos, musics, callback=self.q.put, cancel=self.token, preview=preview).run()
            except BaseException:
                pass
            finally:
                self.q.put({"kind": "finished"})

        self.thread = threading.Thread(target=run, daemon=True, name="amv-job")
        self.thread.start()

    def poll(self, limit: int = 300) -> List[dict]:
        out: List[dict] = []
        for _ in range(limit):
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out

    def cancel(self) -> None:
        self.token.cancel()

    def alive(self) -> bool:
        return self.thread.is_alive()

    def terminate(self) -> None:
        self.token.cancel()


def start_job(settings: EditSettings, videos: List[str], musics: List[str], preview: bool = False):
    """Prefer a separate process (smooth GUI); fall back to a thread if that is impossible."""
    try:
        return JobHandle(settings, videos, musics, preview)
    except Exception:
        return ThreadJob(settings, videos, musics, preview)
