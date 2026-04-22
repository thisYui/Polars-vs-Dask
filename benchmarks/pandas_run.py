"""
benchmarks/pandas_run.py
Standalone Pandas benchmark runner.

Usage:
    python benchmarks/pandas_run.py
    python benchmarks/pandas_run.py --sizes 1M 10M
    python benchmarks/pandas_run.py --sizes 5GB --data-type syn
    python benchmarks/pandas_run.py --workloads filter groupby
"""

import gc
import sys
import time
from pathlib import Path

# ── path setup ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from src.core.config import (
    BENCHMARK_SIZES, SCALABILITY_SIZES, GB_SIZES,
    BENCHMARK_RUNS, WARMUP_RUNS, WORKLOADS,
)
from src.core.benchmark import run_combination
from src.core.timer import format_duration
from src.utils import get_logger, get_dataset_path, print_system_info
from src.workloads import get_workload_fn

logger    = get_logger("pandas_run")
FRAMEWORK = "pandas"

ALL_SIZE_MAP = {**BENCHMARK_SIZES, **SCALABILITY_SIZES, **GB_SIZES}


def _get_n_rows(size_label: str, path: Path) -> int:
    """
    Return row count for a size label.
    - Row-count sizes (1M, 10M, …): read from config dict.
    - GB sizes (1GB, 5GB, …): value is None in config → read from parquet metadata.
    """
    n_rows = ALL_SIZE_MAP.get(size_label)
    if n_rows is not None:
        return n_rows

    # GB-based size: detect row count from parquet metadata (fast, no data load)
    try:
        if path.is_dir():
            import pyarrow.dataset as ds
            n_rows = ds.dataset(path, format="parquet").count_rows()
        else:
            import pyarrow.parquet as pq
            n_rows = pq.read_metadata(path).num_rows
        logger.info(f"  Detected {n_rows:,} rows from parquet metadata")
        return n_rows
    except Exception as exc:
        logger.warning(f"  Could not read row count from parquet ({exc}) — using 0")
        return 0


def main(
    sizes:        list[str],
    workloads:    list[str],
    n_runs:       int = BENCHMARK_RUNS,
    warmup:       int = WARMUP_RUNS,
    results_file: str = "pandas_results.csv",
    file_format:  str = "parquet",
    data_type:    str = "real",
) -> None:
    import pandas as pd

    logger.info("=" * 58)
    logger.info("  PANDAS BENCHMARK")
    logger.info("=" * 58)
    logger.info(f"pandas {pd.__version__} | sizes={sizes} | workloads={workloads}")
    logger.info(f"runs={warmup} warmup + {n_runs} timed")
    logger.info("=" * 58)

    t0 = time.perf_counter()

    for size_label in sizes:
        path = get_dataset_path(size_label, file_format, data_type)

        if not path.exists():
            logger.warning(f"Dataset '{size_label}' not found at {path} — skip")
            continue

        n_rows = _get_n_rows(size_label, path)
        logger.info(f"\n[{size_label}] {n_rows:,} rows")

        for workload in workloads:
            fn = get_workload_fn(FRAMEWORK, workload)
            run_combination(
                fn, path, n_rows, FRAMEWORK, workload, size_label,
                n_runs, warmup, results_file,
            )
            gc.collect()

    logger.info(f"\nPandas done in {format_duration(time.perf_counter()-t0)}")
    logger.info(f"Results → results/raw/{results_file}")


if __name__ == "__main__":
    all_size_choices = list(ALL_SIZE_MAP.keys())

    parser = argparse.ArgumentParser(description="Pandas benchmark runner")
    parser.add_argument("--sizes",     nargs="+", default=["1M", "10M"],
        choices=all_size_choices)
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS, choices=WORKLOADS)
    parser.add_argument("--runs",      type=int,  default=BENCHMARK_RUNS)
    parser.add_argument("--format",    choices=["parquet", "csv"],    default="parquet")
    parser.add_argument("--output",    default="pandas_results.csv")
    parser.add_argument("--sysinfo",   action="store_true")
    parser.add_argument("--data-type", choices=["real", "syn"],       default="real")
    args = parser.parse_args()

    if args.sysinfo:
        print_system_info()

    main(
        sizes=args.sizes,
        workloads=args.workloads,
        n_runs=args.runs,
        results_file=args.output,
        file_format=args.format,
        data_type=args.data_type,
    )