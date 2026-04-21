"""
benchmarks/pandas_run.py
Standalone Pandas benchmark runner.

Usage:
    python benchmarks/pandas_run.py
    python benchmarks/pandas_run.py --sizes 1M 10M
    python benchmarks/pandas_run.py --workloads filter groupby
"""

import gc
import sys
import time
from pathlib import Path

# ── path setup ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import argparse

from src.core.config import BENCHMARK_SIZES, BENCHMARK_RUNS, WARMUP_RUNS, WORKLOADS
from src.core.benchmark import run_combination
from src.core.timer import format_duration
from src.utils import get_logger, get_dataset_path, print_system_info
from src.workloads import get_workload_fn

logger    = get_logger("pandas_run")
FRAMEWORK = "pandas"


def main(
    sizes:        list[str],
    workloads:    list[str],
    n_runs:       int = BENCHMARK_RUNS,
    warmup:       int = WARMUP_RUNS,
    results_file: str = "pandas_results.csv",
    file_format:  str = "parquet",
    data_type: str = "real",
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
        n_rows = BENCHMARK_SIZES.get(size_label) or __import__(
            "core.config", fromlist=["SCALABILITY_SIZES"]
        ).SCALABILITY_SIZES.get(size_label)
        path = get_dataset_path(size_label, file_format, data_type)

        if not path.exists():
            logger.warning(f"Dataset '{size_label}' not found at {path} — skip")
            continue

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
    from src.core.config import SCALABILITY_SIZES
    all_sizes = list({**BENCHMARK_SIZES, **SCALABILITY_SIZES}.keys())

    parser = argparse.ArgumentParser(description="Pandas benchmark runner")
    parser.add_argument("--sizes",     nargs="+", default=["1M", "10M"],
        choices=["1M", "5M", "10M", "50M", "100M",
                 "1GB", "5GB", "10GB", "20GB", "50GB"])
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS,     choices=WORKLOADS)
    parser.add_argument("--runs",      type=int,  default=BENCHMARK_RUNS)
    parser.add_argument("--format",    choices=["parquet", "csv"],        default="parquet")
    parser.add_argument("--output",    default="pandas_results.csv")
    parser.add_argument("--sysinfo",   action="store_true")
    parser.add_argument("--data-type", choices=["real", "syn"], default="real")
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