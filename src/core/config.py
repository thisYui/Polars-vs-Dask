"""
src/core/config.py
Central configuration — single source of truth for all parameters.
Machine: Windows, 16 GB RAM.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────
# Project Root & Directory Layout
# ─────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent   # bigdata-pandas-polars-dask/

DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"          # downloaded Amazon gz/json
PROCESSED_DIR   = DATA_DIR / "processed"   # parquet after clean
SYNTHETIC_DIR   = DATA_DIR / "synthetic"   # fully synthetic datasets
BENCHMARK_DIR   = DATA_DIR / "benchmark"   # ready-to-use splits (1M/10M/100M)

RESULTS_DIR     = ROOT_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TABLES_DIR      = RESULTS_DIR / "tables"
PLOTS_DIR       = RESULTS_DIR / "plots"

LOG_DIR         = ROOT_DIR / "logs"

# Auto-create all directories
for _d in [
    RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, BENCHMARK_DIR,
    RAW_RESULTS_DIR, TABLES_DIR, PLOTS_DIR, LOG_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# Machine Profile
# ─────────────────────────────────────────────────────────
MACHINE_RAM_GB      = 16     # Physical RAM (GB)
RAM_THRESHOLD_GB    = 12     # Datasets above this → "larger than RAM" regime

# ─────────────────────────────────────────────────────────
# Benchmark Split Sizes  (rows)
# ─────────────────────────────────────────────────────────
BENCHMARK_SIZES = {
    "1M":   1_000_000,
    "10M":  10_000_000,
    "100M": 100_000_000,
}

# Extended sizes for scalability analysis
SCALABILITY_SIZES = {
    "1M":   1_000_000,
    "5M":   5_000_000,
    "10M":  10_000_000,
    "20M": 20_000_000,
    "50M":  50_000_000,
    "100M": 100_000_000,
}

# ─────────────────────────────────────────────────────────
# Dataset Schema  (Amazon Reviews)
# ─────────────────────────────────────────────────────────
SCHEMA_COLUMNS = [
    "review_id",
    "user_id",
    "product_id",
    "rating",
    "review_text",
    "review_time",
    "category",
    "verified_purchase",
]

# Real Amazon JSON → internal field mapping
AMAZON_FIELD_MAP = {
    "reviewerID":    "user_id",
    "asin":          "product_id",
    "reviewText":    "review_text",
    "overall":       "rating",
    "unixReviewTime":"review_time",
    "verified":      "verified_purchase",
}

# Synthetic generation cardinality
N_USERS    = 500_000
N_PRODUCTS = 100_000
CATEGORIES = [
    "Electronics", "Books", "Clothing", "Home & Kitchen",
    "Sports", "Toys", "Automotive", "Health", "Beauty", "Tools",
]
RATING_DISTRIBUTION = [0.05, 0.05, 0.10, 0.30, 0.50]   # 1–5 stars

# ─────────────────────────────────────────────────────────
# Benchmark Run Settings
# ─────────────────────────────────────────────────────────
WARMUP_RUNS     = 1     # excluded from measurements
BENCHMARK_RUNS  = 3     # timed repetitions → averaged
TIMEOUT_SECONDS = 600   # per workload (10 min)

# ─────────────────────────────────────────────────────────
# Workload Parameters
# ─────────────────────────────────────────────────────────
WORKLOADS = ["filter", "groupby", "join", "pipeline"]

FILTER_RATING_THRESHOLD = 4          # rating >= 4
GROUPBY_COLUMN          = "product_id"
PRODUCT_METADATA_PATH   = PROCESSED_DIR / "product_metadata.parquet"

# ─────────────────────────────────────────────────────────
# Framework Settings
# ─────────────────────────────────────────────────────────
FRAMEWORKS = ["pandas", "polars", "dask"]

# Pandas
PANDAS_CHUNK_SIZE = 100_000          # rows per chunk when reading CSV

# Polars
POLARS_STREAMING  = True             # enable streaming for > RAM datasets
POLARS_N_THREADS  = None             # None = auto (all cores)

# Dask
DASK_PARTITION_SIZE      = "256MB"
DASK_N_WORKERS           = 4
DASK_THREADS_PER_WORKER  = 2
DASK_MEMORY_LIMIT        = "3GB"     # per worker

# ─────────────────────────────────────────────────────────
# IO Settings
# ─────────────────────────────────────────────────────────
DEFAULT_FORMAT   = "parquet"         # preferred dataset format
PARQUET_COMPRESSION = "snappy"
GENERATOR_CHUNK  = 2_000_000         # rows per generation chunk (~300 MB peak)

# ─────────────────────────────────────────────────────────
# Memory Profiler
# ─────────────────────────────────────────────────────────
MEMORY_POLL_INTERVAL = 0.05          # seconds between RSS polls

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = LOG_DIR / "benchmark.log"