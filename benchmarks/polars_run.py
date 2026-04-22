"""
benchmarks/polars_run.py
Standalone Polars benchmark runner.
Supports lazy (default), eager, or both modes.

Usage:
    python benchmarks/polars_run.py
    python benchmarks/polars_run.py --sizes 1M 10M 100M --mode lazy
    python benchmarks/polars_run.py --sizes 5GB --data-type syn --mode lazy
    python benchmarks/polars_run.py --mode both
"""

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from src.core.config import (
    BENCHMARK_SIZES, SCALABILITY_SIZES, GB_SIZES,
    BENCHMARK_RUNS, WARMUP_RUNS, WORKLOADS,
)
from src.core.benchmark import run_combination
from src.core.timer import format_duration
from src.utils import get_logger, get_dataset_path
from src.workloads import WORKLOAD_REGISTRY

logger   = get_logger("polars_run")

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
    mode:         str = "lazy",   # "lazy" | "eager" | "both"
    n_runs:       int = BENCHMARK_RUNS,
    warmup:       int = WARMUP_RUNS,
    results_file: str = "polars_results.csv",
    file_format:  str = "parquet",
    data_type:    str = "real",
) -> None:
    import polars as pl

    modes = []
    if mode in ("lazy",  "both"): modes.append("polars_lazy")
    if mode in ("eager", "both"): modes.append("polars_eager")

    logger.info("=" * 58)
    logger.info("  POLARS BENCHMARK")
    logger.info("=" * 58)
    logger.info(f"polars {pl.__version__} | mode={mode} | sizes={sizes}")
    logger.info(f"runs={warmup} warmup + {n_runs} timed")
    logger.info("=" * 58)

    t0 = time.perf_counter()

    for size_label in sizes:
        path = get_dataset_path(size_label, file_format, data_type)

        if not path.exists():
            logger.warning(f"Dataset '{size_label}' not found — skip")
            continue

        n_rows = _get_n_rows(size_label, path)
        logger.info(f"\n[{size_label}] {n_rows:,} rows")

        for framework_label in modes:
            for workload in workloads:
                fn = WORKLOAD_REGISTRY[framework_label][workload]
                run_combination(
                    fn, path, n_rows, framework_label, workload, size_label,
                    n_runs, warmup, results_file,
                )
                gc.collect()

    logger.info(f"\nPolars done in {format_duration(time.perf_counter()-t0)}")
    logger.info(f"Results → results/raw/{results_file}")


if __name__ == "__main__":
    all_size_choices = list(ALL_SIZE_MAP.keys())

    parser = argparse.ArgumentParser(description="Polars benchmark runner")
    parser.add_argument("--sizes",     nargs="+", default=["1M", "10M"],
        choices=all_size_choices)
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS, choices=WORKLOADS)
    parser.add_argument("--mode",      choices=["lazy", "eager", "both"], default="lazy")
    parser.add_argument("--runs",      type=int, default=BENCHMARK_RUNS)
    parser.add_argument("--format",    choices=["parquet", "csv"],        default="parquet")
    parser.add_argument("--output",    default="polars_results.csv")
    parser.add_argument("--data-type", choices=["real", "syn"],           default="real")
    args = parser.parse_args()

    main(
        sizes=args.sizes,
        workloads=args.workloads,
        mode=args.mode,
        n_runs=args.runs,
        results_file=args.output,
        file_format=args.format,
        data_type=args.data_type,
    )