"""
utils.py
Shared utilities: logging setup, result persistence, timing helpers.
"""

import csv
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import LOG_LEVEL, LOG_FILE, TABLES_DIR, RAW_RESULTS_DIR


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger writing to file + console."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """
    Context manager that measures wall-clock time.

    Usage:
        with timer("my operation") as t:
            do_something()
        print(t.elapsed)
    """
    class _Timer:
        elapsed: float = 0.0

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start
        if label:
            get_logger("timer").debug(f"{label}: {t.elapsed:.4f}s")


def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


# ─────────────────────────────────────────────
# Result Persistence
# ─────────────────────────────────────────────

def save_result(record: dict[str, Any], filename: str = "benchmark_results.csv") -> Path:
    """
    Append a single benchmark record to the master CSV results file.

    Args:
        record: Dict with keys like framework, workload, size, time_s, memory_mb, etc.
        filename: Target CSV filename inside results/raw/

    Returns:
        Path to the CSV file.
    """
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    path = RAW_RESULTS_DIR / filename

    fieldnames = [
        "timestamp", "framework", "workload", "dataset_size",
        "n_rows", "run_index", "time_s", "peak_memory_mb",
        "throughput_rows_per_s", "status", "notes",
    ]
    # Fill missing keys with None
    for k in fieldnames:
        record.setdefault(k, None)

    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    return path


def load_results(filename: str = "benchmark_results.csv") -> pd.DataFrame:
    """Load all saved benchmark results into a DataFrame."""
    path = RAW_RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_summary_table(df: pd.DataFrame, name: str) -> Path:
    """Save a summary pivot table as both CSV and JSON."""
    csv_path = TABLES_DIR / f"{name}.csv"
    json_path = TABLES_DIR / f"{name}.json"
    df.to_csv(csv_path, index=True)
    df.to_json(json_path, orient="table", indent=2)
    return csv_path


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw results into a summary table.
    Groups by framework + workload + dataset_size, averages timed runs.
    """
    if df.empty:
        return df
    numeric_cols = ["time_s", "peak_memory_mb", "throughput_rows_per_s"]
    available = [c for c in numeric_cols if c in df.columns]
    summary = (
        df[df["status"] == "ok"]
        .groupby(["framework", "workload", "dataset_size", "n_rows"])[available]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


# ─────────────────────────────────────────────
# System Info
# ─────────────────────────────────────────────

def get_system_info() -> dict[str, Any]:
    """Collect basic system metadata for reproducibility."""
    import platform
    import psutil

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(ram_gb, 2),
    }

    try:
        import pandas
        info["pandas_version"] = pandas.__version__
    except ImportError:
        pass
    try:
        import polars
        info["polars_version"] = polars.__version__
    except ImportError:
        pass
    try:
        import dask
        info["dask_version"] = dask.__version__
    except ImportError:
        pass

    return info


def print_system_info() -> None:
    info = get_system_info()
    print("\n" + "=" * 50)
    print("  SYSTEM INFO")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k:<30} {v}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# File Helpers
# ─────────────────────────────────────────────

def get_dataset_path(size_label: str, fmt: str = "parquet") -> Path:
    """Return the expected path for a synthetic dataset of a given size."""
    from config import SYNTHETIC_DIR
    return SYNTHETIC_DIR / f"reviews_{size_label}.{fmt}"


def get_file_size_mb(path: Path) -> float:
    """Return file size in MB."""
    return path.stat().st_size / (1024 ** 2) if path.exists() else 0.0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
