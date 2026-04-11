"""
memory_profiler.py
Background-thread memory monitor that samples peak RSS usage
during a benchmark workload.

Usage:
    with MemoryProfiler() as mp:
        do_heavy_work()
    print(mp.peak_mb)
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import psutil

from config import MEMORY_POLL_INTERVAL


@dataclass
class MemorySnapshot:
    rss_mb: float = 0.0       # Resident Set Size
    vms_mb: float = 0.0       # Virtual Memory Size
    timestamp: float = field(default_factory=time.perf_counter)


class MemoryProfiler:
    """
    Context manager that continuously polls the current process's memory
    in a background thread and records the peak value.

    Attributes:
        peak_mb (float): Peak RSS memory in MB during the monitored block.
        baseline_mb (float): RSS at entry (before workload).
        net_peak_mb (float): peak_mb - baseline_mb.
        samples (list[MemorySnapshot]): All recorded snapshots.
    """

    def __init__(
        self,
        poll_interval: float = MEMORY_POLL_INTERVAL,
        include_children: bool = True,
    ):
        self.poll_interval = poll_interval
        self.include_children = include_children

        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.baseline_mb: float = 0.0
        self.peak_mb: float = 0.0
        self.samples: list[MemorySnapshot] = []

    # ── private ──────────────────────────────

    def _current_rss_mb(self) -> tuple[float, float]:
        """Return (rss_mb, vms_mb) for this process (+ children if enabled)."""
        try:
            mem = self._process.memory_info()
            rss = mem.rss
            vms = mem.vms
            if self.include_children:
                for child in self._process.children(recursive=True):
                    try:
                        cmem = child.memory_info()
                        rss += cmem.rss
                        vms += cmem.vms
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            return rss / (1024 ** 2), vms / (1024 ** 2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0.0

    def _poll(self) -> None:
        """Background polling loop."""
        while not self._stop_event.is_set():
            rss_mb, vms_mb = self._current_rss_mb()
            snap = MemorySnapshot(rss_mb=rss_mb, vms_mb=vms_mb)
            self.samples.append(snap)
            if rss_mb > self.peak_mb:
                self.peak_mb = rss_mb
            self._stop_event.wait(self.poll_interval)

    # ── public API ────────────────────────────

    def start(self) -> "MemoryProfiler":
        self.samples.clear()
        self._stop_event.clear()
        rss_mb, _ = self._current_rss_mb()
        self.baseline_mb = rss_mb
        self.peak_mb = rss_mb
        self._thread = threading.Thread(target=self._poll, daemon=True, name="MemProfiler")
        self._thread.start()
        return self

    def stop(self) -> "MemoryProfiler":
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval * 10)
        return self

    @property
    def net_peak_mb(self) -> float:
        """Memory increase above baseline."""
        return max(0.0, self.peak_mb - self.baseline_mb)

    def summary(self) -> dict:
        return {
            "baseline_mb": round(self.baseline_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "net_peak_mb": round(self.net_peak_mb, 2),
            "n_samples": len(self.samples),
        }

    # ── context manager ───────────────────────

    def __enter__(self) -> "MemoryProfiler":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ─────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────

@contextmanager
def profile_memory(poll_interval: float = MEMORY_POLL_INTERVAL):
    """
    Lightweight context manager returning a MemoryProfiler.

    Usage:
        with profile_memory() as mp:
            heavy_work()
        print(f"Peak: {mp.peak_mb:.1f} MB")
    """
    mp = MemoryProfiler(poll_interval=poll_interval)
    mp.start()
    try:
        yield mp
    finally:
        mp.stop()


# ─────────────────────────────────────────────
# Standalone helper: measure a callable once
# ─────────────────────────────────────────────

def measure_peak_memory(fn, *args, **kwargs) -> tuple[any, float]:
    """
    Run fn(*args, **kwargs), return (result, peak_rss_mb).
    """
    with MemoryProfiler() as mp:
        result = fn(*args, **kwargs)
    return result, mp.peak_mb


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np

    print("Smoke test: allocating ~500 MB array …")
    with MemoryProfiler() as mp:
        arr = np.zeros((500, 1024, 256), dtype=np.float32)  # ~500 MB
        time.sleep(0.3)
        del arr

    print(mp.summary())
