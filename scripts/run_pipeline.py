"""
run_pipeline.py
Master pipeline runner — chạy toàn bộ project từ đầu đến cuối.

Usage:
    python run_pipeline.py                          # chạy full pipeline
    python run_pipeline.py --sizes 1M 10M 20M      # custom sizes
    python run_pipeline.py --skip-download         # bỏ qua download (đã có data)
    python run_pipeline.py --dry-run               # xem lệnh sẽ chạy, không thực thi
    python run_pipeline.py --resume                # bỏ qua bước đã thành công
    python run_pipeline.py --sysinfo               # in system info và thoát

    # Steps test
    cd script
    python run_pipeline.py --sysinfo
    python run_pipeline.py --dry-run --skip-download --sizes 1M
    python run_pipeline.py --skip-download --sizes 1M --groups data
    python run_pipeline.py --skip-download --sizes 1M --groups benchmark

    #  DOWNLOAD + PREPARE REAL DATA
    cd scripts
    python run_pipeline.py --steps download preprocess split_real --sizes 1M 10M 20M

    # RUN BENCHMARK
    python run_pipeline.py --groups benchmark --sizes 1M 10M 20M

    # BENCHMARK WITH PARTITION
    python run_pipeline.py --groups benchmark --sizes 1M 10M 20M --partition

"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent if _HERE.name == "scripts" else _HERE

LOG_DIR     = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RESUME_FILE = ROOT / "logs" / "pipeline_resume.json"
LOG_FILE    = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# PYTHONPATH cho tất cả subprocess — để src.* import được
_ENV = os.environ.copy()
_ENV["PYTHONPATH"] = str(ROOT)

# ─────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

logger = _setup_logger()

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _python(script: str) -> list[str]:
    return [sys.executable, script]

def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"

# ─────────────────────────────────────────────────────────
# Pipeline step definitions
# ─────────────────────────────────────────────────────────
def build_steps(args: argparse.Namespace) -> list[dict]:
    sizes_flag      = args.sizes
    frameworks_flag = args.frameworks
    source_flag     = args.data_source
    split_extra     = ["--partition"] if args.partition else []
    benchmark_extra = ["--generate-data"] if args.generate_data else []

    return [
        # ── GROUP: data ───────────────────────────────────
        {
            "name":        "download",
            "group":       "data",
            "description": "Download Amazon Reviews dataset",
            "required":    False,
            "cmd": _python("src/data/download_amazon.py") + (
                ["--small"] if args.small_download else
                ["--category"] + args.amazon_categories
            ),
        },
        {
            "name":        "preprocess",
            "group":       "data",
            "description": "Preprocess raw Amazon data → Parquet",
            "required":    False,
            "cmd": _python("src/data/preprocess_amazon.py") + ["--metadata"],
        },
        {
            "name":        "split_real",
            "group":       "data",
            "description": "Split processed data → benchmark splits",
            "required":    False,
            "cmd": _python("src/data/split_dataset.py") + [
                "--source", source_flag, "--sizes", *sizes_flag,
            ] + split_extra,
        },
        {
            "name":        "generate_synthetic",
            "group":       "data",
            "description": "Generate synthetic dataset",
            "required":    True,
            "cmd": _python("src/data/data_generator.py") + ["--sizes", *sizes_flag],
        },
        {
            "name":        "split_synthetic",
            "group":       "data",
            "description": "Split synthetic data → benchmark splits",
            "required":    True,
            "cmd": _python("src/data/split_dataset.py") + [
                "--source", "synthetic", "--sizes", *sizes_flag,
            ] + split_extra,
        },

        # ── GROUP: benchmark ──────────────────────────────
        {
            "name":        "benchmark_pandas",
            "group":       "benchmark",
            "description": "Run Pandas benchmark",
            "required":    True,
            "cmd": _python("benchmarks/pandas_run.py") + [
                "--sizes", *sizes_flag, "--output", "pandas_results.csv",
            ],
        },
        {
            "name":        "benchmark_polars",
            "group":       "benchmark",
            "description": "Run Polars benchmark (lazy mode)",
            "required":    True,
            "cmd": _python("benchmarks/polars_run.py") + [
                "--sizes", *sizes_flag, "--mode", "lazy", "--output", "polars_results.csv",
            ],
        },
        {
            "name":        "benchmark_polars_eager",
            "group":       "benchmark",
            "description": "Run Polars benchmark (eager mode)",
            "required":    False,
            "cmd": _python("benchmarks/polars_run.py") + [
                "--sizes", *sizes_flag, "--mode", "eager", "--output", "polars_eager_results.csv",
            ],
        },
        {
            "name":        "benchmark_dask",
            "group":       "benchmark",
            "description": "Run Dask benchmark",
            "required":    True,
            "cmd": _python("benchmarks/dask_run.py") + [
                "--sizes", *sizes_flag,
                "--workers", str(args.dask_workers),
                "--memory-limit", args.dask_memory,
                "--output", "dask_results.csv",
            ],
        },
        {
            "name":        "benchmark_all",
            "group":       "benchmark",
            "description": "Run full benchmark matrix (all frameworks together)",
            "required":    False,
            "cmd": _python("benchmarks/run_all.py") + [
                "--sizes", *sizes_flag, "--frameworks", *frameworks_flag,
                "--output", "all_results.csv",
            ] + benchmark_extra,
        },
    ]


# ─────────────────────────────────────────────────────────
# Resume state helpers
# ─────────────────────────────────────────────────────────
def _load_resume() -> set[str]:
    if RESUME_FILE.exists():
        try:
            return set(json.loads(RESUME_FILE.read_text()).get("completed", []))
        except Exception:
            return set()
    return set()

def _save_resume(completed: set[str]) -> None:
    RESUME_FILE.write_text(
        json.dumps({"completed": sorted(completed), "updated": datetime.now().isoformat()}, indent=2)
    )

def _clear_resume() -> None:
    if RESUME_FILE.exists():
        RESUME_FILE.unlink()
    logger.info("Resume state cleared.")


# ─────────────────────────────────────────────────────────
# Step runner
# ─────────────────────────────────────────────────────────
def run_step(step: dict, dry_run: bool = False, verbose: bool = False) -> bool:
    name = step["name"]
    cmd  = step["cmd"]

    logger.info("")
    logger.info("─" * 60)
    logger.info(f"  STEP : {name}")
    logger.info(f"  DESC : {step['description']}")
    logger.info(f"  CMD  : {' '.join(cmd)}")
    logger.info("─" * 60)

    if dry_run:
        logger.info("  [DRY RUN] — skipping execution")
        return True

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=_ENV,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.STDOUT,
        )
        elapsed = time.perf_counter() - t0
        if not verbose and result.stdout:
            for line in result.stdout.splitlines():
                logger.debug(f"    [subprocess] {line}")
        logger.info(f"  ✓ Done in {_fmt(elapsed)}")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"  ✗ FAILED after {_fmt(elapsed)} — exit code {e.returncode}")
        if e.stdout:
            logger.error("  --- subprocess output ---")
            for line in e.stdout.splitlines()[-30:]:
                logger.error(f"    {line}")
        return False

    except FileNotFoundError:
        logger.error(f"  ✗ Script not found: {cmd[1]}")
        return False

    except KeyboardInterrupt:
        logger.warning("  ⚠ Interrupted by user")
        raise


# ─────────────────────────────────────────────────────────
# System info
# ─────────────────────────────────────────────────────────
def print_sysinfo() -> None:
    try:
        import platform, psutil
        print("\n" + "=" * 55)
        print("  SYSTEM INFO")
        print("=" * 55)
        print(f"  OS              : {platform.system()} {platform.release()}")
        print(f"  Python          : {platform.python_version()}")
        print(f"  CPU logical     : {psutil.cpu_count(logical=True)}")
        print(f"  CPU physical    : {psutil.cpu_count(logical=False)}")
        print(f"  RAM total       : {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"  RAM available   : {psutil.virtual_memory().available / 1024**3:.1f} GB")
        for pkg in ("pandas", "polars", "dask", "pyarrow", "numpy"):
            try:
                mod = __import__(pkg)
                print(f"  {pkg:<16}: {mod.__version__}")
            except ImportError:
                print(f"  {pkg:<16}: not installed")
        print("=" * 55 + "\n")
    except ImportError:
        logger.warning("psutil not installed — run: pip install psutil")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master pipeline runner — Pandas vs Polars vs Dask benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument("--sizes", nargs="+", default=["1M", "10M"],
        choices=["1M", "5M", "10M", "20M", "50M", "100M"])
    data_grp.add_argument("--data-source", choices=["auto", "real", "synthetic"], default="auto")
    data_grp.add_argument("--amazon-categories", nargs="+",
        default=["All_Beauty", "Gift_Cards", "Software"])
    data_grp.add_argument("--small-download", action="store_true", default=True)
    data_grp.add_argument("--partition", action="store_true",
        help="Multi-file partitions cho dataset 20M+")

    bench_grp = parser.add_argument_group("Benchmark")
    bench_grp.add_argument("--frameworks", nargs="+",
        default=["pandas", "polars_lazy", "dask"],
        choices=["pandas", "polars_lazy", "polars_eager", "dask"])
    bench_grp.add_argument("--dask-workers", type=int, default=4)
    bench_grp.add_argument("--dask-memory", default="3GB")
    bench_grp.add_argument("--generate-data", action="store_true")

    ctrl_grp = parser.add_argument_group("Pipeline control")
    ctrl_grp.add_argument("--groups", nargs="+", choices=["data", "benchmark"])
    ctrl_grp.add_argument("--steps", nargs="+")
    ctrl_grp.add_argument("--skip-download", action="store_true")
    ctrl_grp.add_argument("--skip-steps", nargs="+", default=[])
    ctrl_grp.add_argument("--resume", action="store_true")
    ctrl_grp.add_argument("--clear-resume", action="store_true")
    ctrl_grp.add_argument("--dry-run", action="store_true")
    ctrl_grp.add_argument("--verbose", action="store_true",
        help="In toàn bộ output subprocess ra console")
    ctrl_grp.add_argument("--sysinfo", action="store_true")

    args = parser.parse_args()

    if args.sysinfo:
        print_sysinfo()
        sys.exit(0)

    if args.clear_resume:
        _clear_resume()
        sys.exit(0)

    logger.info("=" * 60)
    logger.info("  BIG DATA BENCHMARK PIPELINE")
    logger.info("  Pandas vs Polars vs Dask")
    logger.info("=" * 60)
    logger.info(f"  Root       : {ROOT}")
    logger.info(f"  Sizes      : {args.sizes}")
    logger.info(f"  Frameworks : {args.frameworks}")
    logger.info(f"  Data source: {args.data_source}")
    logger.info(f"  Partition  : {args.partition}")
    logger.info(f"  Dry run    : {args.dry_run}")
    logger.info(f"  Resume     : {args.resume}")
    logger.info(f"  Log file   : {LOG_FILE}")
    logger.info("=" * 60)

    all_steps = build_steps(args)

    # Filter steps
    if args.steps:
        selected = [s for s in all_steps if s["name"] in args.steps]
        unknown  = set(args.steps) - {s["name"] for s in all_steps}
        if unknown:
            logger.warning(f"Unknown step names: {unknown}")
    elif args.groups:
        selected = [s for s in all_steps if s["group"] in args.groups]
    else:
        selected = all_steps

    if args.skip_download:
        selected = [s for s in selected if s["name"] not in {"download", "preprocess", "split_real"}]
    if args.skip_steps:
        selected = [s for s in selected if s["name"] not in set(args.skip_steps)]

    if not selected:
        logger.error("No steps selected.")
        sys.exit(1)

    logger.info(f"\nSteps to run ({len(selected)}):")
    for s in selected:
        logger.info(f"  [{s['group']:<10}] {s['name']:<30} {s['description']}")

    completed = _load_resume() if args.resume else set()
    if completed:
        logger.info(f"\nResuming — already completed: {sorted(completed)}")

    t_pipeline = time.perf_counter()
    results    = {}
    failed     = []

    try:
        for step in selected:
            name = step["name"]

            if args.resume and name in completed:
                logger.info(f"\n  ↷ Skipping '{name}' (already completed)")
                results[name] = True
                continue

            success = run_step(step, dry_run=args.dry_run, verbose=args.verbose)
            results[name] = success

            if success:
                completed.add(name)
                if not args.dry_run:
                    _save_resume(completed)
            else:
                failed.append(name)
                if step["required"]:
                    logger.error(
                        f"\n  Required step '{name}' failed — stopping pipeline.\n"
                        f"  Fix the error then run with --resume to continue."
                    )
                    break
                else:
                    logger.warning(f"  Optional step '{name}' failed — continuing.")

    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted (Ctrl+C)")
        logger.info("Run with --resume to continue.")
        sys.exit(130)

    total_time = time.perf_counter() - t_pipeline
    ok_steps   = [n for n, v in results.items() if v]
    fail_steps = [n for n, v in results.items() if not v]

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total time : {_fmt(total_time)}")
    logger.info(f"  Completed  : {len(ok_steps)}/{len(results)}")
    for n in ok_steps:
        logger.info(f"    ✓ {n}")
    if fail_steps:
        logger.info(f"  Failed     : {len(fail_steps)}")
        for n in fail_steps:
            logger.info(f"    ✗ {n}")
    logger.info(f"  Log        : {LOG_FILE}")
    logger.info("=" * 60)

    sys.exit(0 if not fail_steps else 1)


if __name__ == "__main__":
    main()