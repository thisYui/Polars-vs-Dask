"""
benchmarks/run_all.py
Master runner: executes the full benchmark matrix across all frameworks.

Can be driven in two ways:
  1. Programmatically via ExperimentConfig (called from run_pipeline.py)
  2. CLI (backward-compatible with the original interface)

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --sizes 1M 10M
    python benchmarks/run_all.py --sizes 5GB --data-type syn
    python benchmarks/run_all.py --frameworks pandas polars
    python benchmarks/run_all.py --workloads filter groupby
    python benchmarks/run_all.py --generate-data   # auto-generate datasets first
    python benchmarks/run_all.py --sysinfo         # print system info and exit

    # Config-driven (used by run_pipeline.py):
    python benchmarks/run_all.py --config logical
    python benchmarks/run_all.py --config physical
    python benchmarks/run_all.py --config validation
"""

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from src.core.config import (
    BENCHMARK_SIZES, SCALABILITY_SIZES, GB_SIZES,
    BENCHMARK_RUNS, WARMUP_RUNS,
    WORKLOADS, FRAMEWORKS,
    DASK_N_WORKERS, DASK_THREADS_PER_WORKER, DASK_MEMORY_LIMIT,
)
from src.core.benchmark import BenchmarkRunner
from src.core.timer import format_duration
from src.utils import get_logger, print_system_info, merge_result_files
from src.workloads import WORKLOAD_REGISTRY

logger = get_logger("run_all")

ALL_SIZE_MAP = {**BENCHMARK_SIZES, **SCALABILITY_SIZES, **GB_SIZES}

RESULT_FILES = {
    "pandas":        "pandas_results.csv",
    "polars":        "polars_results.csv",
    "polars_lazy":   "polars_results.csv",
    "polars_eager":  "polars_eager_results.csv",
    "dask":          "dask_results.csv",
}


# ─────────────────────────────────────────────────────────
# Dask cluster helpers
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
# Config snapshot helper
# ─────────────────────────────────────────────────────────

def _save_config_snapshot(cfg_dict: dict) -> None:
    """Persist the run configuration to results/configs/run_config.json."""
    out = Path("results/configs")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg_dict, fh, indent=2, default=str)
    logger.info("Config snapshot → results/configs/run_config.json")


# ─────────────────────────────────────────────────────────
# Core runner  (accepts ExperimentConfig OR raw kwargs)
# ─────────────────────────────────────────────────────────

def main(
    cfg=None,
    *,
    # Legacy keyword args kept for direct-call backward compat
    sizes=None,
    frameworks=None,
    workloads=None,
    n_runs:        int  = BENCHMARK_RUNS,
    warmup:        int  = WARMUP_RUNS,
    file_format:   str  = "parquet",
    results_file:  str  = "all_results.csv",
    dask_workers:  int  = DASK_N_WORKERS,
    dask_threads:  int  = DASK_THREADS_PER_WORKER,
    dask_memory:   str  = DASK_MEMORY_LIMIT,
    generate_data: bool = False,
    data_type:     str  = "real",
) -> None:
    """
    Run the full benchmark matrix.

    Config-driven (recommended):
        from src.core.experiment_config import load_experiment_config
        main(load_experiment_config("logical"))

    Legacy keyword args (backward-compatible):
        main(sizes={"1M": 1_000_000}, frameworks=["pandas"], ...)
    """
    # ── Resolve parameters from cfg or kwargs ─────────────
    if cfg is not None:
        from src.core.experiment_config import ExperimentConfig
        if not isinstance(cfg, ExperimentConfig):
            raise TypeError(f"cfg must be an ExperimentConfig, got {type(cfg)}")

        size_labels = cfg.size_labels()
        sizes_dict  = {lbl: ALL_SIZE_MAP.get(lbl) for lbl in size_labels}
        frameworks  = cfg.frameworks
        workloads   = cfg.workloads
        n_runs      = cfg.repeat
        warmup      = cfg.warmup
        # data_type: use first entry; logical/validation may list both
        data_type   = cfg.data_types[0] if cfg.data_types else "real"

        _snapshot = {**cfg.__dict__, "resolved_size_labels": size_labels}

    else:
        # Legacy path — caller passed explicit kwargs
        if sizes is None:
            raise ValueError("Provide either cfg= or sizes=")
        sizes_dict = sizes
        frameworks = frameworks or list(WORKLOAD_REGISTRY.keys())
        workloads  = workloads  or WORKLOADS
        _snapshot  = dict(
            sizes=list(sizes_dict.keys()),
            frameworks=frameworks,
            workloads=workloads,
            n_runs=n_runs,
            warmup=warmup,
            data_type=data_type,
        )

    t0 = time.perf_counter()

    # ── Config snapshot ───────────────────────────────────
    _save_config_snapshot(_snapshot)

    # ── Optional data generation ──────────────────────────
    if generate_data:
        logger.info("Generating benchmark datasets …")
        from src.data.split_dataset import prepare_benchmark_splits
        prepare_benchmark_splits(source="auto", sizes=sizes_dict)

    # ── Dask cluster ──────────────────────────────────────
    cluster, client = _maybe_start_dask(frameworks, dask_workers, dask_threads, dask_memory)

    try:
        sub_registry = {
            fw: {wl: fn for wl, fn in WORKLOAD_REGISTRY[fw].items() if wl in workloads}
            for fw in frameworks
            if fw in WORKLOAD_REGISTRY
        }

        runner = BenchmarkRunner(
            workload_registry=sub_registry,
            frameworks=frameworks,
            workloads=workloads,
            sizes=sizes_dict,
            n_runs=n_runs,
            warmup=warmup,
            results_file=results_file,
            file_format=file_format,
        )
        runner.run()

    finally:
        _stop_dask(cluster, client)

    # ── Merge individual result files ─────────────────────
    individual_files = [RESULT_FILES[fw] for fw in frameworks if fw in RESULT_FILES]
    merged = merge_result_files(*individual_files, output=results_file)
    if not merged.empty:
        logger.info(f"Merged {len(merged):,} result rows → results/raw/{results_file}")

    logger.info(f"\nTotal wall time: {format_duration(time.perf_counter() - t0)}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_size_choices = list(ALL_SIZE_MAP.keys())
    all_frameworks   = list(WORKLOAD_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Full benchmark matrix: Pandas vs Polars vs Dask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Config-driven (recommended — reads configs/<n>.yaml)
  python benchmarks/run_all.py --config logical
  python benchmarks/run_all.py --config physical

  # Quick manual run
  python benchmarks/run_all.py --sizes 1M

  # Full benchmark
  python benchmarks/run_all.py --sizes 1M 10M 100M

  # GB-based size (synthetic data)
  python benchmarks/run_all.py --sizes 5GB --data-type syn

  # Pandas + Polars only
  python benchmarks/run_all.py --frameworks pandas polars --workloads filter groupby

  # Auto-generate data then benchmark
  python benchmarks/run_all.py --sizes 1M 10M --generate-data
        """,
    )

    # ── Config-driven flag ────────────────────────────────
    parser.add_argument(
        "--config",
        metavar="NAME",
        default=None,
        help="Load experiment from configs/<n>.yaml  (logical | physical | validation). "
             "When set, --sizes / --frameworks / --workloads / --runs / --warmup are ignored.",
    )

    # ── Legacy / manual flags (backward-compat) ───────────
    parser.add_argument("--sizes",        nargs="+", default=["1M", "10M"],
                        choices=all_size_choices)
    parser.add_argument("--frameworks",   nargs="+",
                        default=["pandas", "polars_lazy", "dask"],
                        choices=all_frameworks)
    parser.add_argument("--workloads",    nargs="+", default=WORKLOADS, choices=WORKLOADS)
    parser.add_argument("--runs",         type=int,  default=BENCHMARK_RUNS)
    parser.add_argument("--warmup",       type=int,  default=WARMUP_RUNS)
    parser.add_argument("--format",       choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--output",       default="all_results.csv")
    parser.add_argument("--dask-workers", type=int,  default=DASK_N_WORKERS)
    parser.add_argument("--dask-threads", type=int,  default=DASK_THREADS_PER_WORKER)
    parser.add_argument("--dask-memory",  default=DASK_MEMORY_LIMIT)
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--data-type",    choices=["real", "syn"], default="real")
    parser.add_argument("--sysinfo",      action="store_true",
                        help="Print system info and exit")
    args = parser.parse_args()

    if args.sysinfo:
        print_system_info()
        sys.exit(0)

    # Config-driven path
    if args.config:
        from src.core.experiment_config import load_experiment_config
        _cfg = load_experiment_config(args.config)
        logger.info(f"Loaded experiment config: {_cfg}")
        main(cfg=_cfg, results_file=args.output, file_format=args.format)

    # Legacy manual path
    else:
        _sizes = {k: ALL_SIZE_MAP[k] for k in args.sizes}
        main(
            sizes=_sizes,
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
            data_type=args.data_type,
        )