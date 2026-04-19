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
BENCHMARK_DIR   = DATA_DIR / "benchmark"   # ready-to-use splits (1M/10M/50M)
BENCHMARK_REAL_DIR   = DATA_DIR / "benchmark_real"   # ready-to-use splits (1M/10M/50M)
BENCHMARK_SYN_DIR   = DATA_DIR / "benchmark_syn"   # ready-to-use splits (1M/10M/50M)

RESULTS_DIR     = ROOT_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TABLES_DIR      = RESULTS_DIR / "tables"
PLOTS_DIR       = RESULTS_DIR / "plots"

LOG_DIR         = ROOT_DIR / "logs"

# Auto-create all directories
for _d in [
    RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, BENCHMARK_REAL_DIR,
    BENCHMARK_SYN_DIR, RAW_RESULTS_DIR, TABLES_DIR, PLOTS_DIR, LOG_DIR,
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
    "50M":  50_000_000,
    "100M": 100_000_000,
}

# ─────────────────────────────────────────────────────────
# Dataset Schema  (Amazon Reviews)
# ─────────────────────────────────────────────────────────
SCHEMA_COLUMNS = [
    "review_id",        # R + 10 digits  (synthetic) / hash(user+product) (real)
    "user_id",          # 28-char alphanumeric
    "product_id",       # ASIN  e.g. B096S6LZV4
    "parent_asin",      # parent product ASIN (groups variants)
    "rating",           # int8  1-5
    "review_title",     # short title of the review
    "review_text",      # full review body
    "review_time",      # datetime (normalised to date)
    "helpful_vote",     # int32  number of helpful votes
    "category",         # string  e.g. "Electronics"
    "verified_purchase",# bool
]

# Real Amazon JSON (2023 HuggingFace format) → internal field mapping
# Fields present in the HF dataset:
#   rating, title, text, images, asin, parent_asin,
#   user_id, timestamp (ms), helpful_vote, verified_purchase
AMAZON_FIELD_MAP = {
    # HuggingFace 2023 format
    "asin":          "product_id",
    "parent_asin":   "parent_asin",
    "title":         "review_title",
    "text":          "review_text",
    "timestamp":     "review_time",   # milliseconds epoch → handled in _cast_chunk
    "helpful_vote":  "helpful_vote",
    "verified_purchase": "verified_purchase",
    # Legacy UCSD format (kept for backward compat)
    "reviewerID":    "user_id",
    "reviewText":    "review_text",
    "overall":       "rating",
    "unixReviewTime":"review_time",
    "summary":       "review_title",
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
# ── Memory budget ──────────────────────────────────────────────────────────
# Rule of thumb on a 16 GB machine:
#   Leave ~4 GB for OS + Pandas/Polars overhead.
#   Remaining ~12 GB split across workers.
#   2 workers × 4 GB = 8 GB total managed by Dask  ← safe for join workload
#
# Why the old 2 GB limit caused OOM on join:
#   Dask was trying to persist both the reviews AND metadata inside each
#   worker (combined ~3–4 GB per worker), exceeding the 1.86 GiB RSS limit.
#   Raising the limit AND switching to broadcast join in join.py fixes this.
# ──────────────────────────────────────────────────────────────────────────
DASK_PARTITION_SIZE      = "128MB"
DASK_N_WORKERS           = 2
DASK_THREADS_PER_WORKER  = 2
DASK_MEMORY_LIMIT        = "4GB"     # per worker  (was 2GB → OOM on join)

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