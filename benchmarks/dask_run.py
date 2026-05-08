"""
benchmarks/dask_run.py
Standalone Dask benchmark runner.
Spins up a LocalCluster, runs workloads, tears down cleanly.

Usage:
    python benchmarks/dask_run.py
    python benchmarks/dask_run.py --sizes 10M 100M
    python benchmarks/dask_run.py --sizes 5GB --data-type syn
    python benchmarks/dask_run.py --workers 4 --memory-limit 3GB
    python benchmarks/dask_run.py --no-cluster
    python benchmarks/dask_run.py --partition-tuning
"""

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from src.core.config import (
    BENCHMARK_SIZES, SCALABILITY_SIZES, GB_SIZES, SYNTHETIC_STRESS_SIZES,
    BENCHMARK_RUNS, WARMUP_RUNS, WORKLOADS,
    DASK_N_WORKERS, DASK_THREADS_PER_WORKER,
    DASK_MEMORY_LIMIT, DASK_PARTITION_SIZE,
)
from src.core.benchmark import run_combination, run_single
from src.core.timer import format_duration
from src.utils import get_logger, get_dataset_path, save_result
from src.workloads import get_workload_fn

logger    = get_logger("dask_run")
FRAMEWORK = "dask"

ALL_SIZE_MAP = {**BENCHMARK_SIZES, **SCALABILITY_SIZES, **GB_SIZES, **SYNTHETIC_STRESS_SIZES}


# ─────────────────────────────────────────────────────────
# Row count helper
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# Cluster management
# ─────────────────────────────────────────────────────────

def start_cluster(n_workers, threads, memory_limit):
    try:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads,
            memory_limit=memory_limit,
            silence_logs=True,
        )
        client = Client(cluster)
        logger.info(f"Dask dashboard → {client.dashboard_link}")
        return cluster, client
    except Exception as exc:
        logger.warning(f"LocalCluster failed ({exc}) — using synchronous scheduler")
        return None, None


def stop_cluster(cluster, client):
    import logging
    try:
        if client:
            logging.getLogger("distributed.client").setLevel(logging.ERROR)
            logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
            client.close()
        if cluster:
            cluster.close()
        time.sleep(0.2)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────

def main(
    sizes:          list[str],
    workloads:      list[str],
    n_runs:         int  = BENCHMARK_RUNS,
    warmup:         int  = WARMUP_RUNS,
    results_file:   str  = "dask_results.csv",
    file_format:    str  = "parquet",
    use_cluster:    bool = True,
    n_workers:      int  = DASK_N_WORKERS,
    threads:        int  = DASK_THREADS_PER_WORKER,
    memory_limit:   str  = DASK_MEMORY_LIMIT,
    partition_size: str  = DASK_PARTITION_SIZE,
    data_type:      str  = "real",
) -> None:
    import dask

    logger.info("=" * 58)
    logger.info("  DASK BENCHMARK")
    logger.info("=" * 58)
    logger.info(f"dask {dask.__version__} | cluster={use_cluster}")
    if use_cluster:
        logger.info(f"workers={n_workers} × {threads} threads | mem/worker={memory_limit}")
    logger.info(f"partition_size={partition_size} | sizes={sizes}")
    logger.info(f"runs={warmup} warmup + {n_runs} timed")
    logger.info("=" * 58)

    cluster, client = None, None
    if use_cluster:
        cluster, client = start_cluster(n_workers, threads, memory_limit)

    t0 = time.perf_counter()

    try:
        for size_label in sizes:
            path = get_dataset_path(size_label, file_format, data_type)

            if not path.exists():
                logger.warning(f"Dataset '{size_label}' not found — skip")
                continue

            n_rows = _get_n_rows(size_label, path)
            logger.info(f"\n[{size_label}] {n_rows:,} rows")

            for workload in workloads:
                fn = get_workload_fn(FRAMEWORK, workload)
                run_combination(
                    fn, path, n_rows, FRAMEWORK, workload, size_label,
                    n_runs, warmup, results_file,
                    extra_notes=f"partition_size={partition_size}",
                )
                if client:
                    client.run(gc.collect)

    finally:
        stop_cluster(cluster, client)

    logger.info(f"\nDask done in {format_duration(time.perf_counter()-t0)}")
    logger.info(f"Results → results/raw/{results_file}")


# ─────────────────────────────────────────────────────────
# Partition tuning experiment
# ─────────────────────────────────────────────────────────

def partition_tuning(
    size_label:      str       = "10M",
    workload:        str       = "groupby",
    partition_sizes: list[str] = None,
    results_file:    str       = "dask_partition_tuning.csv",
) -> None:
    """Benchmark Dask across multiple partition sizes to find the optimum."""
    partition_sizes = partition_sizes or ["64MB", "128MB", "256MB", "512MB", "1GB"]

    path   = get_dataset_path(size_label)
    n_rows = _get_n_rows(size_label, path)
    fn     = get_workload_fn(FRAMEWORK, workload)

    logger.info(f"\nPartition tuning | {size_label} | {workload}")
    for psize in partition_sizes:
        logger.info(f"  partition_size={psize}")
        for run_idx in range(1, BENCHMARK_RUNS + 1):
            rec = run_single(
                fn, path, n_rows,
                f"dask_{psize}", workload, size_label, run_idx,
                extra_notes=f"partition_size={psize}",
            )
            save_result(rec, results_file)
        gc.collect()


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_size_choices = list(ALL_SIZE_MAP.keys())

    parser = argparse.ArgumentParser(description="Dask benchmark runner")
    parser.add_argument("--sizes",     nargs="+", default=["1M", "10M"],
        choices=all_size_choices,
        metavar="SIZE",
        help=f"Dataset size labels. Choices: {all_size_choices}")
    parser.add_argument("--workloads",      nargs="+", default=WORKLOADS,     choices=WORKLOADS)
    parser.add_argument("--runs",           type=int,  default=BENCHMARK_RUNS)
    parser.add_argument("--format",         choices=["parquet", "csv"],        default="parquet")
    parser.add_argument("--output",         default="dask_results.csv")
    parser.add_argument("--workers",        type=int,  default=DASK_N_WORKERS)
    parser.add_argument("--threads",        type=int,  default=DASK_THREADS_PER_WORKER)
    parser.add_argument("--memory-limit",   default=DASK_MEMORY_LIMIT)
    parser.add_argument("--partition-size", default=DASK_PARTITION_SIZE)
    parser.add_argument("--no-cluster",     action="store_true")
    parser.add_argument("--partition-tuning", action="store_true",
                        help="Run partition size tuning experiment")
    parser.add_argument("--data-type", choices=["real", "syn"], default="real")
    args = parser.parse_args()

    if args.partition_tuning:
        partition_tuning()
    else:
        main(
            sizes=args.sizes,
            workloads=args.workloads,
            n_runs=args.runs,
            results_file=args.output,
            file_format=args.format,
            use_cluster=not args.no_cluster,
            n_workers=args.workers,
            threads=args.threads,
            memory_limit=args.memory_limit,
            partition_size=args.partition_size,
            data_type=args.data_type,
        )