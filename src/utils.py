"""
src/utils.py
Shared helpers: logging, result I/O, path resolution, system info.
"""

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import (
    LOG_LEVEL, LOG_FILE,
    RAW_RESULTS_DIR, TABLES_DIR,
    BENCHMARK_DIR, SYNTHETIC_DIR,
)


# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────

def get_dataset_path(size_label: str, fmt: str = "parquet") -> Path:
    # Ưu tiên folder partition (dành cho 20M+)
    folder = BENCHMARK_DIR / size_label
    if folder.exists() and any(folder.glob("part-*.parquet")):
        return folder
    # Fallback single file
    return BENCHMARK_DIR / f"reviews_{size_label}.{fmt}"


def get_synthetic_path(size_label: str, fmt: str = "parquet") -> Path:
    return SYNTHETIC_DIR / f"reviews_{size_label}.{fmt}"


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024**2 if path.exists() else 0.0


# ─────────────────────────────────────────────────────────
# Result persistence
# ─────────────────────────────────────────────────────────

_RESULT_FIELDS = [
    "timestamp", "framework", "workload", "dataset_size",
    "n_rows", "run_index", "time_s", "peak_memory_mb",
    "throughput_rows_per_s", "status", "notes",
]


def save_result(record: dict[str, Any], filename: str = "benchmark_results.csv") -> Path:
    """Append one benchmark record to a CSV file in results/raw/."""
    record = dict(record)
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    for k in _RESULT_FIELDS:
        record.setdefault(k, None)

    path = RAW_RESULTS_DIR / filename
    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_RESULT_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(record)

    return path


def load_results(filename: str = "benchmark_results.csv") -> pd.DataFrame:
    path = RAW_RESULTS_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_summary_table(df: pd.DataFrame, name: str) -> Path:
    csv_path  = TABLES_DIR / f"{name}.csv"
    json_path = TABLES_DIR / f"{name}.json"
    df.to_csv(csv_path, index=True)
    df.to_json(json_path, orient="table", indent=2)
    return csv_path


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric = [c for c in ["time_s", "peak_memory_mb", "throughput_rows_per_s"]
               if c in df.columns]
    summary = (
        df[df["status"] == "ok"]
        .groupby(["framework", "workload", "dataset_size", "n_rows"])[numeric]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def merge_result_files(*filenames: str, output: str = "all_results.csv") -> pd.DataFrame:
    """Concatenate multiple per-framework result CSVs into one master file."""
    frames = [load_results(f) for f in filenames if load_results(f).shape[0] > 0]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(RAW_RESULTS_DIR / output, index=False)
    return merged


# ─────────────────────────────────────────────────────────
# System info
# ─────────────────────────────────────────────────────────

def get_system_info() -> dict[str, Any]:
    import platform
    import psutil

    info = {
        "os":               platform.system(),
        "python_version":   platform.python_version(),
        "cpu_logical":      psutil.cpu_count(logical=True),
        "cpu_physical":     psutil.cpu_count(logical=False),
        "ram_total_gb":     round(psutil.virtual_memory().total / 1024**3, 2),
    }
    for pkg in ("pandas", "polars", "dask", "pyarrow", "numpy"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = mod.__version__
        except ImportError:
            pass
    return info


def print_system_info() -> None:
    info = get_system_info()
    print("\n" + "=" * 52)
    print("  SYSTEM INFO")
    print("=" * 52)
    for k, v in info.items():
        print(f"  {k:<28} {v}")
    print("=" * 52 + "\n")