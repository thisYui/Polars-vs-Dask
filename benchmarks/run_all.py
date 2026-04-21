"""
benchmarks/run_all.py
Master runner: executes the full benchmark matrix across all frameworks.

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --sizes 1M 10M
    python benchmarks/run_all.py --frameworks pandas polars
    python benchmarks/run_all.py --workloads filter groupby
    python benchmarks/run_all.py --generate-data   # auto-generate datasets first
    python benchmarks/run_all.py --sysinfo         # print system info and exit
"""

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from src.core.config import (
    BENCHMARK_SIZES, SCALABILITY_SIZES,
    BENCHMARK_RUNS, WARMUP_RUNS,
    WORKLOADS, FRAMEWORKS,
    DASK_N_WORKERS, DASK_THREADS_PER_WORKER, DASK_MEMORY_LIMIT,
)
from src.core.benchmark import BenchmarkRunner
from src.core.timer import format_duration
from src.utils import get_logger, print_system_info, merge_result_files
from src.workloads import WORKLOAD_REGISTRY

logger = get_logger("run_all")

# Per-framework result filenames
RESULT_FILES = {
    "pandas":       "pandas_results.csv",
    "polars":       "polars_results.csv",
    "polars_lazy":  "polars_results.csv",
    "polars_eager": "polars_eager_results.csv",
    "dask":         "dask_results.csv",
}


# ─────────────────────────────────────────────────────────
# Optional: start Dask cluster for the run
# ─────────────────────────────────────────────────────────

def _maybe_start_dask(frameworks, n_workers, threads, memory_limit):
    if "dask" not in frameworks:
        return None, None
    try:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads,
            memory_limit=memory_limit,
            silence_logs=True,
        )
        client = Client(cluster)
        logger.info(f"Dask cluster started — dashboard: {client.dashboard_link}")
        return cluster, client
    except Exception as exc:
        logger.warning(f"Could not start Dask cluster ({exc}) — using synchronous scheduler")
        return None, None


def _stop_dask(cluster, client):
    try:
        if client:  client.close()
        if cluster: cluster.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main(
    sizes:        dict[str, int],
    frameworks:   list[str],
    workloads:    list[str],
    n_runs:       int  = BENCHMARK_RUNS,
    warmup:       int  = WARMUP_RUNS,
    file_format:  str  = "parquet",
    results_file: str  = "all_results.csv",
    dask_workers: int  = DASK_N_WORKERS,
    dask_threads: int  = DASK_THREADS_PER_WORKER,
    dask_memory:  str  = DASK_MEMORY_LIMIT,
    generate_data: bool = False,
) -> None:
    t0 = time.perf_counter()

    # ── Optional data generation ──────────────────────────
    if generate_data:
        logger.info("Generating benchmark datasets …")
        from src.data.split_dataset import prepare_benchmark_splits
        prepare_benchmark_splits(source="auto", sizes=sizes)

    # ── Dask cluster ──────────────────────────────────────
    cluster, client = _maybe_start_dask(frameworks, dask_workers, dask_threads, dask_memory)

    try:
        # ── Build per-framework sub-registries ────────────
        # Filter registry to only requested frameworks & workloads
        sub_registry = {
            fw: {wl: fn for wl, fn in WORKLOAD_REGISTRY[fw].items() if wl in workloads}
            for fw in frameworks
            if fw in WORKLOAD_REGISTRY
        }

        runner = BenchmarkRunner(
            workload_registry=sub_registry,
            frameworks=frameworks,
            workloads=workloads,
            sizes=sizes,
            n_runs=n_runs,
            warmup=warmup,
            results_file=results_file,
            file_format=file_format,
        )
        runner.run()

    finally:
        _stop_dask(cluster, client)

    # ── Merge individual files if they exist ──────────────
    individual_files = [RESULT_FILES[fw] for fw in frameworks if fw in RESULT_FILES]
    merged = merge_result_files(*individual_files, output=results_file)
    if not merged.empty:
        logger.info(f"Merged {len(merged):,} result rows → results/raw/{results_file}")

    logger.info(f"\nTotal wall time: {format_duration(time.perf_counter() - t0)}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_size_map  = {**BENCHMARK_SIZES, **SCALABILITY_SIZES}
    all_frameworks = list(WORKLOAD_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Full benchmark matrix: Pandas vs Polars vs Dask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run (1M rows, all frameworks, all workloads)
  python benchmarks/run_all.py --sizes 1M

  # Full benchmark
  python benchmarks/run_all.py --sizes 1M 10M 100M

  # Pandas + Polars only, filter and groupby only
  python benchmarks/run_all.py --frameworks pandas polars --workloads filter groupby

  # Generate data automatically then benchmark
  python benchmarks/run_all.py --sizes 1M 10M --generate-data
        """,
    )
    parser.add_argument("--sizes",     nargs="+", default=["1M", "10M"],
        choices=["1M", "5M", "10M", "50M", "100M",
                 "1GB", "5GB", "10GB", "20GB", "50GB"])
    parser.add_argument("--frameworks",  nargs="+", default=["pandas", "polars_lazy", "dask"],
                        choices=all_frameworks)
    parser.add_argument("--workloads",   nargs="+", default=WORKLOADS,
                        choices=WORKLOADS)
    parser.add_argument("--runs",        type=int,  default=BENCHMARK_RUNS)
    parser.add_argument("--warmup",      type=int,  default=WARMUP_RUNS)
    parser.add_argument("--format",      choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--output",      default="all_results.csv")
    parser.add_argument("--dask-workers",type=int,  default=DASK_N_WORKERS)
    parser.add_argument("--dask-threads",type=int,  default=DASK_THREADS_PER_WORKER)
    parser.add_argument("--dask-memory", default=DASK_MEMORY_LIMIT)
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate datasets before running benchmarks")
    parser.add_argument("--sysinfo",     action="store_true",
                        help="Print system info and exit")
    args = parser.parse_args()

    if args.sysinfo:
        print_system_info()
        sys.exit(0)

    sizes = {k: all_size_map[k] for k in args.sizes}

    main(
        sizes=sizes,
        frameworks=args.frameworks,
        workloads=args.workloads,
        n_runs=args.runs,
        warmup=args.warmup,
        file_format=args.format,
        results_file=args.output,
        dask_workers=args.dask_workers,
        dask_threads=args.dask_threads,
        dask_memory=args.dask_memory,
        generate_data=args.generate_data,
    )