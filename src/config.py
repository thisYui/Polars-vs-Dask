"""
config.py
Central configuration for Big Data Benchmark: Pandas vs Polars vs Dask
Machine: Windows, 16GB RAM
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create dirs if missing
for _dir in [RAW_DIR, SYNTHETIC_DIR, PROCESSED_DIR,
             RAW_RESULTS_DIR, TABLES_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Machine Profile
# ─────────────────────────────────────────────
MACHINE_RAM_GB = 16          # Physical RAM in GB
RAM_THRESHOLD_GB = 12        # Datasets above this are "larger than RAM" experiments
RAM_THRESHOLD_ROWS = 80_000_000  # ~12-14GB depending on schema

# ─────────────────────────────────────────────
# Dataset Sizes (rows)
# ─────────────────────────────────────────────
DATASET_SIZES = {
    "small":   1_000_000,    # ~150 MB  – all frameworks fast
    "medium":  5_000_000,    # ~750 MB  – Polars advantage shows
    "large":  10_000_000,    # ~1.5 GB  – Polars dominates
    "xlarge": 50_000_000,    # ~7.5 GB  – pushing Pandas limit
    "huge":  100_000_000,    # ~15 GB   – larger than RAM → Dask only viable
}

# Quick-run subset (skip huge for fast testing)
FAST_SIZES = {k: v for k, v in DATASET_SIZES.items() if k in ("small", "medium", "large")}

# ─────────────────────────────────────────────
# Dataset Schema  (Amazon Reviews)
# ─────────────────────────────────────────────
COLUMNS = [
    "review_id",
    "user_id",
    "product_id",
    "rating",
    "review_text",
    "review_time",
    "category",
    "verified_purchase",
]

# Cardinality for synthetic generation
N_USERS = 500_000
N_PRODUCTS = 100_000
CATEGORIES = [
    "Electronics", "Books", "Clothing", "Home & Kitchen",
    "Sports", "Toys", "Automotive", "Health", "Beauty", "Tools",
]
RATING_DISTRIBUTION = [0.05, 0.05, 0.10, 0.30, 0.50]  # 1-5 stars (skewed positive)

# ─────────────────────────────────────────────
# Benchmark Settings
# ─────────────────────────────────────────────
WARMUP_RUNS = 1       # Runs before measurement (cache warm-up)
BENCHMARK_RUNS = 3    # Timed repetitions → results averaged
TIMEOUT_SECONDS = 600 # Per workload timeout (10 min)

# ─────────────────────────────────────────────
# Workloads
# ─────────────────────────────────────────────
WORKLOADS = ["filter", "groupby", "join", "pipeline"]

# Filter workload
FILTER_RATING_THRESHOLD = 4  # rating >= 4

# GroupBy workload
GROUPBY_COLUMN = "product_id"
GROUPBY_AGG = {"rating": ["mean", "count", "sum"]}

# Join workload
PRODUCT_METADATA_PATH = PROCESSED_DIR / "product_metadata.parquet"

# ─────────────────────────────────────────────
# Frameworks
# ─────────────────────────────────────────────
FRAMEWORKS = ["pandas", "polars", "dask"]

# ─────────────────────────────────────────────
# Dask Settings
# ─────────────────────────────────────────────
DASK_PARTITION_SIZE = "256MB"   # Default partition size
DASK_N_WORKERS = 4              # Local cluster workers (adjust to CPU cores)
DASK_THREADS_PER_WORKER = 2
DASK_MEMORY_LIMIT = "3GB"       # Per worker memory limit

# ─────────────────────────────────────────────
# Polars Settings
# ─────────────────────────────────────────────
POLARS_STREAMING = True         # Use streaming for datasets > RAM
POLARS_N_THREADS = None         # None = auto (uses all cores)

# ─────────────────────────────────────────────
# IO Settings
# ─────────────────────────────────────────────
DEFAULT_FILE_FORMAT = "parquet"     # "parquet" or "csv"
RESULTS_FORMAT = "csv"              # Benchmark results saved as CSV
CHUNK_SIZE_CSV = 100_000            # Rows per chunk when reading CSV in Pandas

# ─────────────────────────────────────────────
# Memory Profiling
# ─────────────────────────────────────────────
MEMORY_POLL_INTERVAL = 0.05     # Seconds between memory polls
MEMORY_UNIT = "MB"              # "MB" or "GB"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = ROOT_DIR / "benchmark.log"
