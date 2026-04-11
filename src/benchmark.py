"""
benchmark.py
Core benchmarking engine.

Responsibilities:
  - Warm-up run (excluded from measurements)
  - N timed repetitions per workload
  - Concurrent memory profiling
  - Timeout guard
  - Result persistence via utils.save_result
  - Graceful OOM / crash handling (records status="oom" or "error")
"""

import gc
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import psutil

from config import (
    BENCHMARK_RUNS,
    DATASET_SIZES,
    FRAMEWORKS,
    TIMEOUT_SECONDS,
    WARMUP_RUNS,
    WORKLOADS,
)
from memory_profiler import MemoryProfiler
from utils import (
    get_logger,
    get_dataset_path,
    save_result,
    format_duration,
    build_summary,
    save_summary_table,
    load_results,
)
from workloads import get_workload_fn

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Timeout (POSIX only; Windows fallback below)
# ─────────────────────────────────────────────

class TimeoutError(Exception):
    pass


@contextmanager
def _timeout(seconds: int):
    """Timeout context manager. Uses SIGALRM on Unix; threading-based on Windows."""
    if sys.platform == "win32":
        # Windows: use a threading.Timer to raise from another thread
        import threading
        timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutError()))
        # Simple flag approach instead
        yield  # Windows timeout not enforced at signal level; rely on Dask/OS limits
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Exceeded {seconds}s timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ─────────────────────────────────────────────
# Single workload runner
# ─────────────────────────────────────────────

def run_single(
    fn: Callable,
    dataset_path: Path,
    n_rows: int,
    framework: str,
    workload: str,
    size_label: str,
    run_index: int,
) -> dict:
    """
    Execute one timed + memory-profiled run of a workload function.

    Returns a result record dict.
    """
    record = {
        "framework": framework,
        "workload": workload,
        "dataset_size": size_label,
        "n_rows": n_rows,
        "run_index": run_index,
        "status": "ok",
        "time_s": None,
        "peak_memory_mb": None,
        "throughput_rows_per_s": None,
        "notes": "",
    }

    try:
        mp = MemoryProfiler()
        mp.start()
        t0 = time.perf_counter()

        fn(dataset_path)  # execute workload

        elapsed = time.perf_counter() - t0
        mp.stop()

        record["time_s"] = round(elapsed, 4)
        record["peak_memory_mb"] = round(mp.peak_mb, 2)
        record["throughput_rows_per_s"] = round(n_rows / elapsed) if elapsed > 0 else 0

        logger.info(
            f"  [{framework:<7}] {workload:<10} {size_label:<8} "
            f"run={run_index} | {elapsed:.2f}s | {mp.peak_mb:.0f} MB"
        )

    except MemoryError:
        mp.stop()
        record["status"] = "oom"
        record["notes"] = "MemoryError – dataset likely exceeded available RAM"
        logger.warning(
            f"  [{framework:<7}] {workload:<10} {size_label:<8} → OOM"
        )

    except TimeoutError:
        mp.stop()
        record["status"] = "timeout"
        record["notes"] = f"Exceeded {TIMEOUT_SECONDS}s timeout"
        logger.warning(
            f"  [{framework:<7}] {workload:<10} {size_label:<8} → TIMEOUT"
        )

    except Exception as exc:
        mp.stop()
        record["status"] = "error"
        record["notes"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.error(
            f"  [{framework:<7}] {workload:<10} {size_label:<8} → ERROR: {exc}"
        )
        logger.debug(traceback.format_exc())

    finally:
        # Aggressive cleanup between runs
        gc.collect()

    return record


# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────

class BenchmarkRunner:
    """
    Orchestrates the full benchmark matrix:
        frameworks × workloads × dataset_sizes × BENCHMARK_RUNS

    Args:
        frameworks: List of frameworks to test (default: all).
        workloads: List of workloads to run (default: all).
        sizes: Dict of {label: n_rows} to test (default: DATASET_SIZES).
        n_runs: Number of timed runs per combination.
        warmup: Number of warm-up runs (excluded from results).
        results_file: Filename for persisting results.
        file_format: Dataset file format to read ("parquet" or "csv").
        skip_missing: If True, skip size if dataset file doesn't exist.
    """

    def __init__(
        self,
        frameworks: list[str] = None,
        workloads: list[str] = None,
        sizes: dict[str, int] = None,
        n_runs: int = BENCHMARK_RUNS,
        warmup: int = WARMUP_RUNS,
        results_file: str = "benchmark_results.csv",
        file_format: str = "parquet",
        skip_missing: bool = True,
    ):
        self.frameworks = frameworks or FRAMEWORKS
        self.workloads = workloads or WORKLOADS
        self.sizes = sizes or DATASET_SIZES
        self.n_runs = n_runs
        self.warmup = warmup
        self.results_file = results_file
        self.file_format = file_format
        self.skip_missing = skip_missing

    def run(self) -> None:
        """Execute the full benchmark matrix."""
        logger.info("=" * 60)
        logger.info("  BIG DATA BENCHMARK: Pandas vs Polars vs Dask")
        logger.info("=" * 60)
        logger.info(f"Frameworks : {self.frameworks}")
        logger.info(f"Workloads  : {self.workloads}")
        logger.info(f"Sizes      : {list(self.sizes)}")
        logger.info(f"Runs/combo : {self.warmup} warmup + {self.n_runs} measured")
        logger.info("=" * 60)

        total_combos = (
            len(self.frameworks) * len(self.workloads) * len(self.sizes)
        )
        done = 0
        t_total = time.perf_counter()

        for size_label, n_rows in self.sizes.items():
            dataset_path = get_dataset_path(size_label, self.file_format)

            if not dataset_path.exists():
                if self.skip_missing:
                    logger.warning(
                        f"Dataset '{size_label}' not found at {dataset_path} – skipping. "
                        f"Run data_generator.py first."
                    )
                    continue
                else:
                    raise FileNotFoundError(dataset_path)

            logger.info(f"\n{'─'*60}")
            logger.info(f"Dataset: {size_label} ({n_rows:,} rows) | {dataset_path.name}")
            logger.info(f"{'─'*60}")

            for framework in self.frameworks:
                for workload in self.workloads:
                    fn = get_workload_fn(framework, workload)
                    self._run_combination(
                        fn, dataset_path, n_rows, framework, workload, size_label
                    )
                    done += 1
                    pct = done / total_combos * 100
                    elapsed_total = time.perf_counter() - t_total
                    logger.info(
                        f"Progress: {done}/{total_combos} ({pct:.0f}%) | "
                        f"Total elapsed: {format_duration(elapsed_total)}"
                    )

        logger.info("\n" + "=" * 60)
        logger.info(f"Benchmark complete in {format_duration(time.perf_counter() - t_total)}")
        logger.info("=" * 60)
        self._save_summary()

    def _run_combination(
        self,
        fn: Callable,
        dataset_path: Path,
        n_rows: int,
        framework: str,
        workload: str,
        size_label: str,
    ) -> None:
        """Run warm-up + timed repetitions for a single fw/workload/size combo."""
        label = f"[{framework}/{workload}/{size_label}]"

        # ── Warm-up ──
        for w in range(self.warmup):
            logger.info(f"{label} warm-up {w+1}/{self.warmup} …")
            try:
                fn(dataset_path)
            except Exception as e:
                logger.warning(f"{label} warm-up failed: {e} – skipping timed runs")
                _save_error_record(framework, workload, size_label, n_rows, str(e),
                                   self.results_file)
                return
            gc.collect()

        # ── Timed runs ──
        for run_idx in range(1, self.n_runs + 1):
            record = run_single(
                fn, dataset_path, n_rows,
                framework, workload, size_label, run_idx
            )
            save_result(record, self.results_file)

            # Stop repeating if we already got OOM / timeout
            if record["status"] in ("oom", "timeout"):
                for remaining in range(run_idx + 1, self.n_runs + 1):
                    save_result(
                        {**record, "run_index": remaining,
                         "notes": f"Skipped after {record['status']} on run {run_idx}"},
                        self.results_file,
                    )
                break

    def _save_summary(self) -> None:
        """Build and persist summary tables after benchmark completes."""
        df = load_results(self.results_file)
        if df.empty:
            logger.warning("No results to summarize.")
            return

        summary = build_summary(df)
        path = save_summary_table(summary, "benchmark_summary")
        logger.info(f"Summary saved → {path}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _save_error_record(framework, workload, size_label, n_rows, msg, results_file):
    for i in range(1, BENCHMARK_RUNS + 1):
        save_result({
            "framework": framework, "workload": workload,
            "dataset_size": size_label, "n_rows": n_rows,
            "run_index": i, "status": "error", "notes": msg[:200],
        }, results_file)


# ─────────────────────────────────────────────
# Additional Experiment Runners
# ─────────────────────────────────────────────

def run_lazy_vs_eager_experiment(
    size_label: str = "medium",
    workloads: list[str] = None,
    results_file: str = "lazy_vs_eager.csv",
) -> None:
    """
    Compare Polars lazy vs eager execution for each workload.
    """
    import polars as pl
    from workloads import (
        run_polars_filter, run_polars_groupby,
        run_polars_join, run_polars_pipeline,
    )

    fns_lazy = {
        "filter": lambda p: run_polars_filter(p, lazy=True),
        "groupby": lambda p: run_polars_groupby(p, lazy=True),
        "join": lambda p: run_polars_join(p, lazy=True),
        "pipeline": lambda p: run_polars_pipeline(p, lazy=True),
    }
    fns_eager = {
        "filter": lambda p: run_polars_filter(p, lazy=False),
        "groupby": lambda p: run_polars_groupby(p, lazy=False),
        "join": lambda p: run_polars_join(p, lazy=False),
        "pipeline": lambda p: run_polars_pipeline(p, lazy=False),
    }

    wl = workloads or WORKLOADS
    path = get_dataset_path(size_label)
    n_rows = DATASET_SIZES[size_label]

    for mode, fns in [("lazy", fns_lazy), ("eager", fns_eager)]:
        for workload in wl:
            for run_idx in range(1, BENCHMARK_RUNS + 1):
                record = run_single(
                    fns[workload], path, n_rows,
                    f"polars_{mode}", workload, size_label, run_idx,
                )
                save_result(record, results_file)


def run_io_experiment(
    size_label: str = "medium",
    results_file: str = "io_benchmark.csv",
) -> None:
    """
    Compare CSV vs Parquet I/O speed for all frameworks.
    """
    import pandas as pd
    import polars as pl
    import dask.dataframe as dd

    n_rows = DATASET_SIZES[size_label]

    io_fns = {
        "pandas_parquet":   lambda p: pd.read_parquet(p),
        "pandas_csv":       lambda p: pd.read_csv(p),
        "polars_parquet":   lambda p: pl.read_parquet(p),
        "polars_csv":       lambda p: pl.read_csv(p),
        "dask_parquet":     lambda p: dd.read_parquet(p).compute(),
        "dask_csv":         lambda p: dd.read_csv(p).compute(),
    }

    for label, fn in io_fns.items():
        framework, fmt = label.rsplit("_", 1)
        path = get_dataset_path(size_label, fmt)
        if not path.exists():
            logger.warning(f"IO experiment: {path} not found, skipping.")
            continue
        for run_idx in range(1, BENCHMARK_RUNS + 1):
            record = run_single(fn, path, n_rows, label, "io_read", size_label, run_idx)
            save_result(record, results_file)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run big data benchmarks")
    parser.add_argument("--frameworks", nargs="+", default=FRAMEWORKS,
                        choices=FRAMEWORKS, help="Frameworks to benchmark")
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS,
                        choices=WORKLOADS, help="Workloads to run")
    parser.add_argument("--sizes", nargs="+",
                        default=["small", "medium", "large"],
                        choices=list(DATASET_SIZES.keys()),
                        help="Dataset sizes to test")
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS,
                        help="Timed runs per combination")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--experiment", choices=["main", "lazy_eager", "io"],
                        default="main", help="Which experiment to run")
    args = parser.parse_args()

    if args.experiment == "main":
        runner = BenchmarkRunner(
            frameworks=args.frameworks,
            workloads=args.workloads,
            sizes={k: DATASET_SIZES[k] for k in args.sizes},
            n_runs=args.runs,
            file_format=args.format,
        )
        runner.run()

    elif args.experiment == "lazy_eager":
        run_lazy_vs_eager_experiment()

    elif args.experiment == "io":
        run_io_experiment()
