"""
src/core/benchmark.py
Core benchmark engine.

Handles: warmup runs, N timed repetitions, concurrent memory profiling,
graceful OOM/timeout/error capture, and result persistence.
"""

import gc
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from src.core.config import (
    BENCHMARK_RUNS, TIMEOUT_SECONDS, WARMUP_RUNS,
    WORKLOADS, FRAMEWORKS, BENCHMARK_SIZES,
)
from src.core.timer import format_duration
from src.core.memory_profiler import MemoryProfiler
from src.utils import get_logger, save_result, load_results, build_summary, save_summary_table

logger = get_logger("core.benchmark")


# ─────────────────────────────────────────────────────────
# Single timed run
# ─────────────────────────────────────────────────────────

def run_single(
    fn:           Callable,
    dataset_path: Path,
    n_rows:       int,
    framework:    str,
    workload:     str,
    size_label:   str,
    run_index:    int,
    extra_notes:  str = "",
) -> dict:
    """
    Execute one timed + memory-profiled call of fn(dataset_path).
    Returns a result record dict ready for save_result().
    """
    record = dict(
        framework=framework,
        workload=workload,
        dataset_size=size_label,
        n_rows=n_rows,
        run_index=run_index,
        status="ok",
        time_s=None,
        peak_memory_mb=None,
        throughput_rows_per_s=None,
        notes=extra_notes,
    )

    mp = MemoryProfiler()
    try:
        mp.start()
        t0 = time.perf_counter()
        fn(dataset_path)
        elapsed = time.perf_counter() - t0
        mp.stop()

        record["time_s"]               = round(elapsed, 4)
        record["peak_memory_mb"]       = round(mp.peak_mb, 2)
        record["throughput_rows_per_s"] = round(n_rows / elapsed) if elapsed > 0 else 0

        logger.info(
            f"  [{framework:<14}] {workload:<10} {size_label:<6} "
            f"run={run_index} | {elapsed:.2f}s | {mp.peak_mb:.0f} MB"
        )

    except MemoryError:
        mp.stop()
        record["status"] = "oom"
        record["notes"]  = "MemoryError – dataset exceeded available RAM"
        logger.warning(f"  [{framework}] {workload}/{size_label} run={run_index} → OOM")

    except Exception as exc:
        mp.stop()
        record["status"] = "error"
        record["notes"]  = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.error(f"  [{framework}] {workload}/{size_label} ERROR: {exc}")
        logger.debug(traceback.format_exc())

    finally:
        gc.collect()

    return record


# ─────────────────────────────────────────────────────────
# Combination runner (warmup + N timed runs)
# ─────────────────────────────────────────────────────────

def run_combination(
    fn:           Callable,
    dataset_path: Path,
    n_rows:       int,
    framework:    str,
    workload:     str,
    size_label:   str,
    n_runs:       int       = BENCHMARK_RUNS,
    warmup:       int       = WARMUP_RUNS,
    results_file: str       = "benchmark_results.csv",
    extra_notes:  str       = "",
) -> list[dict]:
    """
    Warmup → N timed runs for one (framework, workload, size) combination.
    Saves each record immediately so no results are lost on crash.
    Returns list of result records.
    """
    label = f"[{framework}/{workload}/{size_label}]"
    records = []

    # ── Warm-up ──────────────────────────────────────────
    for w in range(warmup):
        logger.info(f"  {label} warm-up {w+1}/{warmup}")
        try:
            fn(dataset_path)
        except Exception as exc:
            logger.warning(f"  {label} warm-up failed: {exc} — skipping timed runs")
            for i in range(1, n_runs + 1):
                rec = dict(
                    framework=framework, workload=workload,
                    dataset_size=size_label, n_rows=n_rows,
                    run_index=i, status="error",
                    notes=f"warmup failed: {str(exc)[:150]}",
                )
                save_result(rec, results_file)
                records.append(rec)
            return records
        gc.collect()

    # ── Timed runs ───────────────────────────────────────
    for run_idx in range(1, n_runs + 1):
        rec = run_single(
            fn, dataset_path, n_rows,
            framework, workload, size_label, run_idx, extra_notes,
        )
        save_result(rec, results_file)
        records.append(rec)

        # If run failed, fill remaining slots and stop
        if rec["status"] in ("oom", "timeout", "error"):
            for skip_idx in range(run_idx + 1, n_runs + 1):
                skipped = {**rec, "run_index": skip_idx,
                           "notes": f"skipped after {rec['status']} on run {run_idx}"}
                save_result(skipped, results_file)
                records.append(skipped)
            break

    return records


# ─────────────────────────────────────────────────────────
# Full benchmark matrix runner
# ─────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Orchestrates the full matrix:
        frameworks × workloads × dataset_sizes × BENCHMARK_RUNS

    Args:
        workload_registry : dict mapping framework → {workload: fn}
        frameworks        : subset of frameworks to run
        workloads         : subset of workloads to run
        sizes             : dict {label: n_rows}
        n_runs            : timed repetitions per combo
        warmup            : warm-up runs (excluded)
        results_file      : filename inside results/raw/
        file_format       : "parquet" or "csv"
        skip_missing      : skip if dataset file absent
    """

    def __init__(
        self,
        workload_registry: dict,
        frameworks:    list[str] = None,
        workloads:     list[str] = None,
        sizes:         dict      = None,
        n_runs:        int       = BENCHMARK_RUNS,
        warmup:        int       = WARMUP_RUNS,
        results_file:  str       = "benchmark_results.csv",
        file_format:   str       = "parquet",
        skip_missing:  bool      = True,
    ):
        self.registry     = workload_registry
        self.frameworks   = frameworks or FRAMEWORKS
        self.workloads    = workloads  or WORKLOADS
        self.sizes        = sizes      or BENCHMARK_SIZES
        self.n_runs       = n_runs
        self.warmup       = warmup
        self.results_file = results_file
        self.file_format  = file_format
        self.skip_missing = skip_missing

    def run(self) -> None:
        from src.utils import get_dataset_path
        logger.info("=" * 65)
        logger.info("  BIG DATA BENCHMARK — Pandas vs Polars vs Dask")
        logger.info("=" * 65)
        logger.info(f"Frameworks : {self.frameworks}")
        logger.info(f"Workloads  : {self.workloads}")
        logger.info(f"Sizes      : {list(self.sizes)}")
        logger.info(f"Runs/combo : {self.warmup} warmup + {self.n_runs} timed")
        logger.info("=" * 65)

        total = len(self.frameworks) * len(self.workloads) * len(self.sizes)
        done  = 0
        t0    = time.perf_counter()

        for size_label, n_rows in self.sizes.items():
            path = get_dataset_path(size_label, self.file_format)
            if not path.exists():
                if self.skip_missing:
                    logger.warning(f"Missing dataset '{size_label}': {path} — skipped")
                    continue
                raise FileNotFoundError(path)

            logger.info(f"\n{'─'*65}")
            logger.info(f"  Dataset: {size_label}  ({n_rows:,} rows)  {path.name}")
            logger.info(f"{'─'*65}")

            for framework in self.frameworks:
                if framework not in self.registry:
                    logger.warning(f"Framework '{framework}' not in registry — skipped")
                    continue
                for workload in self.workloads:
                    if workload not in self.registry[framework]:
                        logger.warning(f"Workload '{workload}' missing for {framework}")
                        continue

                    fn = self.registry[framework][workload]
                    run_combination(
                        fn, path, n_rows, framework, workload, size_label,
                        self.n_runs, self.warmup, self.results_file,
                    )
                    done += 1
                    logger.info(
                        f"  Progress {done}/{total} "
                        f"({done/total*100:.0f}%) | "
                        f"elapsed {format_duration(time.perf_counter()-t0)}"
                    )

        logger.info(f"\nDone in {format_duration(time.perf_counter()-t0)}")
        self._finalize()

    def _finalize(self) -> None:
        df = load_results(self.results_file)
        if df.empty:
            return
        summary = build_summary(df)
        path = save_summary_table(summary, "benchmark_summary")
        logger.info(f"Summary → {path}")