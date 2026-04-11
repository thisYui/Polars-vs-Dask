"""
src/core/timer.py
Lightweight timing utilities for benchmark runs.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Any


# ─────────────────────────────────────────────────────────
# Duration formatting
# ─────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable string.

    Examples:
        0.004  → "4.0 ms"
        1.23   → "1.23 s"
        75.0   → "1m 15s"
        3661.0 → "1h 01m 01s"
    """
    if seconds < 0:
        return "0.0 ms"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    if seconds < 3600.0:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


# ─────────────────────────────────────────────────────────
# Context-manager timer
# ─────────────────────────────────────────────────────────

@dataclass
class Timer:
    """
    Simple wall-clock timer usable as a context manager or manually.

    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed_s, t.formatted)

        t = Timer()
        t.start()
        do_work()
        t.stop()
        print(t.elapsed_s)
    """
    label: str = ""
    _start: float = field(default=0.0, repr=False, init=False)
    _end:   float = field(default=0.0, repr=False, init=False)
    _running: bool = field(default=False, repr=False, init=False)

    def start(self) -> "Timer":
        self._start   = time.perf_counter()
        self._running = True
        return self

    def stop(self) -> "Timer":
        self._end     = time.perf_counter()
        self._running = False
        return self

    @property
    def elapsed_s(self) -> float:
        if self._running:
            return time.perf_counter() - self._start
        return self._end - self._start

    @property
    def formatted(self) -> str:
        return format_duration(self.elapsed_s)

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        label = f"[{self.label}] " if self.label else ""
        return f"Timer({label}{self.formatted})"


# ─────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────

@contextmanager
def timed(label: str = ""):
    """Context manager that prints elapsed time on exit."""
    t = Timer(label=label).start()
    try:
        yield t
    finally:
        t.stop()
        tag = f"[{label}] " if label else ""
        print(f"{tag}elapsed: {t.formatted}")


def time_it(fn: Callable, *args, repeat: int = 1, **kwargs) -> tuple[Any, float]:
    """
    Run fn(*args, **kwargs) `repeat` times and return
    (last_result, mean_elapsed_seconds).
    """
    times  = []
    result = None
    for _ in range(repeat):
        t0     = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, sum(times) / len(times)