"""
run_pipeline.py
Master pipeline runner — chạy toàn bộ project từ đầu đến cuối.

Available steps:
    download            Download Amazon Reviews từ Hugging Face (stream, dừng sớm theo --sizes)
    compress            Nén .jsonl → .jsonl.gz trong data/raw/ rồi xóa file gốc
    preprocess          Đọc .jsonl.gz → clean → lưu Parquet (single-file hoặc partitioned)
    split_real          Cắt processed parquet → benchmark splits (1M/10M/...)
    generate_synthetic  Tạo synthetic data bằng numpy (không cần internet)
    split_synthetic     Cắt synthetic data → benchmark splits
    benchmark_pandas    Chạy benchmark Pandas
    benchmark_polars    Chạy benchmark Polars (lazy)
    benchmark_polars_eager  Chạy benchmark Polars (eager)
    benchmark_dask      Chạy benchmark Dask
    benchmark_all       Chạy toàn bộ benchmark matrix

Usage:
    python run_pipeline.py --sysinfo
    python run_pipeline.py --dry-run --sizes 1M
    python run_pipeline.py --resume
    python run_pipeline.py --clear-resume

    # Real data
    python run_pipeline.py --steps download preprocess split_real --sizes 1M 10M
    python run_pipeline.py --steps compress preprocess split_real --sizes 1M 10M 50M --partition

    # Synthetic
    python run_pipeline.py --steps generate_synthetic --sizes 1M 10M
    python run_pipeline.py --steps generate_synthetic split_synthetic --sizes 1M 10M 50M 100M

    # RAM-targeted synthetic (recommended)
    python run_pipeline.py --steps generate_synthetic --target-ram-gb 0.3
    python run_pipeline.py --steps generate_synthetic split_synthetic --target-ram-gb 5 10 20

    # Benchmark
    python run_pipeline.py --groups benchmark --sizes 1M 10M
    python run_pipeline.py --steps benchmark_pandas --sizes 1M 10M

# Main
    python run_pipeline.py --steps compress --partition --verbose
    python run_pipeline.py --steps preprocess --partition --verbose
    python run_pipeline.py --steps split_real --sizes 1M 10M 50M --data-type real --partition --verbose
    python run_pipeline.py --steps generate_synthetic split_synthetic --sizes 1M 10M 50M --data-type syn --partition --verbose
    python run_pipeline.py --steps generate_synthetic --target-ram-gb 5 10 20 --data-type syn --verbose
    python run_pipeline.py --steps generate_synthetic split_synthetic --sizes 1M 10M 50M --data-type syn --partition

    python run_pipeline.py --groups benchmark --sizes 1M 10M --data-type real --partition --verbose
    python run_pipeline.py --groups benchmark --sizes 1M 10M --data-type syn --partition --verbose
    python run_pipeline.py --steps benchmark_dask --sizes 1M --data-type real --partition --verbose
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

def _has_jsonl_files(raw_dir: Path) -> bool:
    return any(raw_dir.glob("*.jsonl"))


# ─────────────────────────────────────────────────────────
# Pipeline step definitions
# ─────────────────────────────────────────────────────────
def build_steps(args: argparse.Namespace) -> list[dict]:
    sizes_flag      = args.sizes
    frameworks_flag = args.frameworks
    source_flag     = args.data_source
    raw_format_flag = args.raw_format
    benchmark_extra = ["--generate-data"] if args.generate_data else []
    data_type_flag  = getattr(args, "data_type", "real")

    preprocess_extra = ["--partition"] if args.partition else []
    split_extra      = ["--partition"] if args.partition else []

    raw_format_extra = (
        ["--raw-format", raw_format_flag] if raw_format_flag != "auto" else []
    )

    _SIZE_MAP = {
        "1M": 1_000_000, "5M": 5_000_000, "10M": 10_000_000,
        "50M": 50_000_000, "100M": 100_000_000,
        # GB labels — row count unknown at pipeline build time;
        # use rough estimate (10M rows/GB) only for computing _download_cap
        "1GB":  10_000_000,  "5GB":  50_000_000,
        "10GB": 100_000_000, "20GB": 200_000_000, "50GB": 500_000_000,
    }
    _max_needed   = max(_SIZE_MAP.get(s, 0) for s in sizes_flag)
    _download_cap = int(_max_needed * 1.2)

    if args.small_download:
        _dl_flag = ["--small"]
    elif args.amazon_categories:
        _dl_flag = ["--category"] + args.amazon_categories
    elif _max_needed <= 2_000_000:
        _dl_flag = ["--small"]
    elif _max_needed <= 20_000_000:
        _dl_flag = ["--category",
                    "Home_and_Kitchen", "Sports_and_Outdoors", "Automotive",
                    "Health_and_Household", "Toys_and_Games", "Office_Products"]
    else:
        _dl_flag = ["--category",
                    "Books", "Electronics", "Clothing_Shoes_and_Jewelry",
                    "Home_and_Kitchen", "Sports_and_Outdoors", "Automotive",
                    "Health_and_Household", "Movies_and_TV", "Toys_and_Games"]

    _compress_workers = min(2, len(list((ROOT / "data" / "raw").glob("*.jsonl"))) or 2)
    _target_mb_extra  = ["--target-file-mb", str(args.target_file_mb)]

    # ── FIX #1: --force and --target-ram-gb both propagated to data_generator ──
    gen_force_flag = ["--force"] if args.force else []
    gen_partition_flag = ["--partition"] if args.partition else []

    if args.target_ram_gb:
        # Generate + auto-split into benchmark_syn/<XGB>/ in one step.
        # --split-benchmark tells data_generator to call _split_to_benchmark_syn
        # internally, so we do NOT need a separate split_synthetic step.
        gen_cmd = _python("src/data/data_generator.py") + [
            "--target-ram-gb", *map(str, args.target_ram_gb),
            "--split-benchmark",
            "--target-file-mb", str(args.target_file_mb),
        ] + gen_partition_flag + gen_force_flag
        # No separate split step needed — generator handles it
        split_syn_cmd = None
    else:
        gen_cmd = _python("src/data/data_generator.py") + [
            "--sizes", *sizes_flag,
        ] + gen_force_flag

        # ── FIX #2: split_synthetic skipped only when target_ram_gb AND no sizes ──
        split_syn_cmd = (
            _python("src/data/split_dataset.py") + [
                "--source", "synthetic",
                "--sizes", *sizes_flag,
            ] + split_extra + _target_mb_extra
        )

    _steps = [
        # ===== DATA =====
        {
            "name":        "download",
            "group":       "data",
            "description": "Download Amazon Reviews dataset từ Hugging Face",
            "required":    False,
            "cmd": _python("src/data/download_amazon.py") + _dl_flag
                   + ["--max-rows", str(_download_cap)],
        },
        {
            "name":        "compress",
            "group":       "data",
            "description": "Nén .jsonl → .jsonl.gz",
            "required":    False,
            "cmd": _python("src/data/compress_jsonl.py") + [
                "--input",   str(ROOT / "data" / "raw"),
                "--workers", str(_compress_workers),
            ],
        },
        {
            "name":        "preprocess",
            "group":       "data",
            "description": "Preprocess raw → Parquet",
            "required":    False,
            "cmd": _python("src/data/preprocess_amazon.py")
                   + ["--metadata"]
                   + preprocess_extra
                   + (["--raw-format", "gz"] if not raw_format_extra else raw_format_extra),
        },
        {
            "name":        "split_real",
            "group":       "data",
            "description": "Split real data",
            "required":    False,
            "cmd": _python("src/data/split_dataset.py") + [
                "--source", "real",
                "--sizes",  *sizes_flag,
            ] + split_extra + _target_mb_extra,
        },
        {
            "name":        "generate_synthetic",
            "group":       "data",
            "description": "Generate synthetic dataset",
            "required":    True,
            "cmd":         gen_cmd,
        },

        *([
            {
                "name":        "split_synthetic",
                "group":       "data",
                "description": "Split synthetic data",
                "required":    True,
                "cmd":         split_syn_cmd,
            }
        ] if split_syn_cmd else []),

        # ===== BENCHMARK =====
        # _result_suffix: use "size" when running GB-label sizes so output files
        # are named pandas_size_results.csv etc. (separate from row-count runs).
        # Otherwise use data_type_flag ("syn" / "real").
        # Note: _result_suffix is computed inline via a lambda-like trick so it
        # stays inside the list literal. We pre-compute it just before.
    ]

    # Pre-compute result suffix (must be outside the list literal)
    _is_gb_run    = args.target_ram_gb or any(s.endswith("GB") for s in sizes_flag)
    _result_suffix = "size" if _is_gb_run else data_type_flag

    return _steps + [
        # ===== BENCHMARK =====
        {
            "name":        "benchmark_pandas",
            "group":       "benchmark",
            "description": "Run Pandas benchmark",
            "required":    True,
            "cmd": _python("benchmarks/pandas_run.py") + [
                "--sizes",     *sizes_flag,
                "--data-type", data_type_flag,
                "--output",    f"pandas_{_result_suffix}_results.csv",
            ],
        },
        {
            "name":        "benchmark_polars",
            "group":       "benchmark",
            "description": "Run Polars (lazy)",
            "required":    True,
            "cmd": _python("benchmarks/polars_run.py") + [
                "--sizes",     *sizes_flag,
                "--data-type", data_type_flag,
                "--mode",      "lazy",
                "--output",    f"polars_{_result_suffix}_results.csv",
            ],
        },
        {
            "name":        "benchmark_polars_eager",
            "group":       "benchmark",
            "description": "Run Polars (eager)",
            "required":    False,
            "cmd": _python("benchmarks/polars_run.py") + [
                "--sizes",     *sizes_flag,
                "--data-type", data_type_flag,
                "--mode",      "eager",
                "--output",    f"polars_eager_{_result_suffix}_results.csv",
            ],
        },
        {
            "name":        "benchmark_dask",
            "group":       "benchmark",
            "description": "Run Dask benchmark",
            "required":    True,
            "cmd": _python("benchmarks/dask_run.py") + [
                "--sizes",        *sizes_flag,
                "--data-type",    data_type_flag,
                "--workers",      str(args.dask_workers),
                "--memory-limit", args.dask_memory,
                "--output",       f"dask_{_result_suffix}_results.csv",
            ],
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
        json.dumps(
            {"completed": sorted(completed), "updated": datetime.now().isoformat()},
            indent=2,
        )
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
    logger.info(f"  CMD  : {' '.join(str(c) for c in cmd)}")
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
        logger.error(f"Script not found: {cmd[1]}")
        return False

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise


# ─────────────────────────────────────────────────────────
# System info
# ─────────────────────────────────────────────────────────
def print_sysinfo() -> None:
    try:
        import platform
        import psutil
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
        choices=["1M", "5M", "10M", "50M", "100M",
                 "1GB", "5GB", "10GB", "20GB", "50GB"])
    data_grp.add_argument("--data-source", choices=["auto", "real", "synthetic"], default="auto")
    data_grp.add_argument("--amazon-categories", nargs="+", default=None)
    data_grp.add_argument("--small-download", action="store_true", default=False)
    data_grp.add_argument(
        "--raw-format", choices=["auto", "gz", "jsonl"], default="auto",
    )
    data_grp.add_argument("--partition", action="store_true")
    data_grp.add_argument("--keep-original", action="store_true", default=False)
    data_grp.add_argument("--target-file-mb", type=int, default=128)
    data_grp.add_argument(
        "--data-type",
        choices=["real", "syn", "both"],
        default="real",
    )
    data_grp.add_argument(
        "--target-ram-gb",
        nargs="+",
        type=float,
        metavar="GB",
        help="Generate synthetic datasets targeting RAM size instead of row count",
    )

    bench_grp = parser.add_argument_group("Benchmark")
    bench_grp.add_argument("--frameworks", nargs="+",
        default=["pandas", "polars_lazy", "dask"],
        choices=["pandas", "polars_lazy", "polars_eager", "dask"])
    bench_grp.add_argument("--dask-workers",  type=int, default=2)
    bench_grp.add_argument("--dask-memory",   default="4GB")
    bench_grp.add_argument("--generate-data", action="store_true")

    ctrl_grp = parser.add_argument_group("Pipeline control")
    ctrl_grp.add_argument("--groups",       nargs="+", choices=["data", "benchmark"])
    ctrl_grp.add_argument("--steps",        nargs="+")
    ctrl_grp.add_argument("--skip-download", action="store_true")
    ctrl_grp.add_argument("--skip-steps",   nargs="+", default=[])
    ctrl_grp.add_argument("--resume",       action="store_true")
    ctrl_grp.add_argument("--clear-resume", action="store_true")
    ctrl_grp.add_argument("--dry-run",      action="store_true")
    ctrl_grp.add_argument("--verbose",      action="store_true")
    ctrl_grp.add_argument("--sysinfo",      action="store_true")
    ctrl_grp.add_argument(
        "--force",
        action="store_true",
        help="Re-generate files even if they already exist (passed to data_generator.py)",
    )

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
    logger.info(f"  Root           : {ROOT}")
    logger.info(f"  Sizes          : {args.sizes}")
    logger.info(f"  Frameworks     : {args.frameworks}")
    logger.info(f"  Data source    : {args.data_source}")
    logger.info(f"  Raw format     : {args.raw_format}")
    logger.info(f"  Partition      : {args.partition}")
    logger.info(f"  Target MB/file : {args.target_file_mb}")
    logger.info(f"  Keep original  : {args.keep_original}")
    logger.info(f"  Force          : {args.force}")          # FIX #3: log it
    logger.info(f"  Target RAM GB  : {args.target_ram_gb}")
    logger.info(f"  Dry run        : {args.dry_run}")
    logger.info(f"  Resume         : {args.resume}")
    logger.info(f"  Log file       : {LOG_FILE}")
    logger.info("=" * 60)

    all_steps = build_steps(args)

    # Inject --keep-original into compress step if requested
    for step in all_steps:
        if step["name"] == "compress" and args.keep_original:
            step["cmd"].append("--keep-original")
            step["description"] += " [giữ .jsonl gốc]"

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