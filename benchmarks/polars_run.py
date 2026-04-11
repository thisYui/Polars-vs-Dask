"""
benchmarks/polars_run.py
Standalone Polars benchmark runner.
Supports lazy (default), eager, or both modes.

Usage:
    python benchmarks/polars_run.py
    python benchmarks/polars_run.py --sizes 1M 10M 100M --mode lazy
    python benchmarks/polars_run.py --mode both
"""

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import argparse

from src.core.config import BENCHMARK_SIZES, BENCHMARK_RUNS, WARMUP_RUNS, WORKLOADS
from src.core.benchmark import run_combination
from src.core.timer import format_duration
from src.utils import get_logger, get_dataset_path
from src.workloads import WORKLOAD_REGISTRY

logger    = get_logger("polars_run")


def main(
    sizes:        list[str],
    workloads:    list[str],
    mode:         str = "lazy",   # "lazy" | "eager" | "both"
    n_runs:       int = BENCHMARK_RUNS,
    warmup:       int = WARMUP_RUNS,
    results_file: str = "polars_results.csv",
    file_format:  str = "parquet",
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

    from core.config import SCALABILITY_SIZES
    all_size_map = {**BENCHMARK_SIZES, **SCALABILITY_SIZES}

    t0 = time.perf_counter()

    for size_label in sizes:
        n_rows = all_size_map.get(size_label)
        path   = get_dataset_path(size_label, file_format)

        if not path.exists():
            logger.warning(f"Dataset '{size_label}' not found — skip")
            continue

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
    from src.core.config import SCALABILITY_SIZES
    all_sizes = list({**BENCHMARK_SIZES, **SCALABILITY_SIZES}.keys())

    parser = argparse.ArgumentParser(description="Polars benchmark runner")
    parser.add_argument("--sizes",     nargs="+", default=["1M", "10M"], choices=all_sizes)
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS,     choices=WORKLOADS)
    parser.add_argument("--mode",      choices=["lazy", "eager", "both"], default="lazy")
    parser.add_argument("--runs",      type=int, default=BENCHMARK_RUNS)
    parser.add_argument("--format",    choices=["parquet", "csv"],        default="parquet")
    parser.add_argument("--output",    default="polars_results.csv")
    args = parser.parse_args()

    main(
        sizes=args.sizes,
        workloads=args.workloads,
        mode=args.mode,
        n_runs=args.runs,
        results_file=args.output,
        file_format=args.format,
    )