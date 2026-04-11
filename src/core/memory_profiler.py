"""
src/core/memory_profiler.py
Background-thread RSS memory monitor.
Tracks peak memory usage during a benchmark workload.

Usage:
    with MemoryProfiler() as mp:
        heavy_work()
    print(mp.peak_mb, mp.net_peak_mb)

    # or as a one-liner:
    result, peak_mb = measure_peak_memory(my_fn, arg1, arg2)
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil

from src.core.config import MEMORY_POLL_INTERVAL


@dataclass
class MemorySnapshot:
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    ts: float = field(default_factory=time.perf_counter)


class MemoryProfiler:
    """
    Polls the current process's RSS in a daemon thread every
    `poll_interval` seconds. Records peak and all samples.

    Attributes:
        baseline_mb : RSS at context-manager entry
        peak_mb     : highest RSS observed during the block
        net_peak_mb : peak_mb - baseline_mb  (memory added by workload)
        samples     : list[MemorySnapshot]
    """

    def __init__(
        self,
        poll_interval: float = MEMORY_POLL_INTERVAL,
        include_children: bool = True,
    ):
        self.poll_interval    = poll_interval
        self.include_children = include_children

        self._proc        = psutil.Process()
        self._stop        = threading.Event()
        self._thread      = None

        self.baseline_mb: float = 0.0
        self.peak_mb:     float = 0.0
        self.samples:     list[MemorySnapshot] = []

    # ── internal ──────────────────────────────────────────

    def _rss(self) -> tuple[float, float]:
        try:
            m = self._proc.memory_info()
            rss, vms = m.rss, m.vms
            if self.include_children:
                for child in self._proc.children(recursive=True):
                    try:
                        cm = child.memory_info()
                        rss += cm.rss
                        vms += cm.vms
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            return rss / 1024**2, vms / 1024**2
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0.0

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            rss_mb, vms_mb = self._rss()
            self.samples.append(MemorySnapshot(rss_mb=rss_mb, vms_mb=vms_mb))
            if rss_mb > self.peak_mb:
                self.peak_mb = rss_mb
            self._stop.wait(self.poll_interval)

    # ── public API ────────────────────────────────────────

    def start(self) -> "MemoryProfiler":
        self.samples.clear()
        self._stop.clear()
        rss_mb, _ = self._rss()
        self.baseline_mb = rss_mb
        self.peak_mb     = rss_mb
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="MemProfiler"
        )
        self._thread.start()
        return self

    def stop(self) -> "MemoryProfiler":
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval * 10)
        return self

    @property
    def net_peak_mb(self) -> float:
        return max(0.0, self.peak_mb - self.baseline_mb)

    def summary(self) -> dict:
        return {
            "baseline_mb":  round(self.baseline_mb, 2),
            "peak_mb":      round(self.peak_mb, 2),
            "net_peak_mb":  round(self.net_peak_mb, 2),
            "n_samples":    len(self.samples),
        }

    # ── context manager ───────────────────────────────────

    def __enter__(self) -> "MemoryProfiler":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ── Convenience helpers ───────────────────────────────────

@contextmanager
def profile_memory(poll_interval: float = MEMORY_POLL_INTERVAL):
    """Thin context manager wrapper."""
    mp = MemoryProfiler(poll_interval=poll_interval)
    mp.start()
    try:
        yield mp
    finally:
        mp.stop()


def measure_peak_memory(fn, *args, **kwargs) -> tuple:
    """Run fn(*args, **kwargs) and return (result, peak_rss_mb)."""
    with MemoryProfiler() as mp:
        result = fn(*args, **kwargs)
    return result, mp.peak_mb