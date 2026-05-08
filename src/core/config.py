"""
src/core/config.py

Central configuration — single source of truth for all project parameters.

This version is calibrated from the real Amazon Reviews 1M sample.

Calibration basis:
- Dataset: real/1M
- Item key: parent_asin
- Rows: 1,000,000
- Unique users: 119,926
- Unique parent_asin items: 476,112
- Rating distribution: empirical from real data
- Text length: log-normal reference fitted from real data

Text length calibration notes (from 01b_calibrate_synthetic.ipynb):
- TEXT_MIN_LEN = p5  of real text_len = 12  chars  (loại bỏ review quá ngắn)
- TEXT_MAX_LEN = p85 of real text_len = 374 chars  (kiểm soát RAM, tránh extreme outlier)
- TEXT_GENERATOR_MAX_LEN = p99 of real text_len = 1_234 chars for realistic validation
- Log-normal distribution là model phù hợp (skewness ≈ 4.7, kurtosis ≈ 55)
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────
# Project Root & Directory Layout
# ─────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent.parent   # bigdata-pandas-polars-dask/

DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"                 # downloaded Amazon gz/json
PROCESSED_DIR = DATA_DIR / "processed"     # parquet after clean
SYNTHETIC_DIR = DATA_DIR / "synthetic"     # fully synthetic datasets

BENCHMARK_DIR = DATA_DIR / "benchmark"
BENCHMARK_REAL_DIR = DATA_DIR / "benchmark_real"
BENCHMARK_SYN_DIR = DATA_DIR / "benchmark_syn"

RESULTS_DIR = ROOT_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

LOG_DIR = ROOT_DIR / "logs"

for _d in [
    RAW_DIR,
    PROCESSED_DIR,
    SYNTHETIC_DIR,
    BENCHMARK_REAL_DIR,
    BENCHMARK_SYN_DIR,
    RAW_RESULTS_DIR,
    TABLES_DIR,
    PLOTS_DIR,
    LOG_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# Machine Profile
# ─────────────────────────────────────────────────────────

MACHINE_RAM_GB = 16
RAM_THRESHOLD_GB = 12


# ─────────────────────────────────────────────────────────
# Benchmark Split Sizes
# ─────────────────────────────────────────────────────────

BENCHMARK_SIZES = {
    "1M": 1_000_000,
    "10M": 10_000_000,
    "100M": 100_000_000,
}

SCALABILITY_SIZES = {
    "1M": 1_000_000,
    "5M": 5_000_000,
    "10M": 10_000_000,
    "50M": 50_000_000,
    "100M": 100_000_000,
}

GB_SIZES: dict[str, None] = {
    "1GB": None,
    "5GB": None,
    "10GB": None,
    "20GB": None,
    "50GB": None,
}

# Stress-test sizes for the three synthetic generation directions.
# Row count is None → resolved at runtime from parquet metadata.
SYNTHETIC_STRESS_SIZES: dict[str, int | None] = {
    "10M_skewed":  None,   # Hướng 2: parent_asin Zipf α=3.0, hot-key skew
    "10M_highuid": None,   # Hướng 3: ~50% unique user_id, high-cardinality
    "100M":        100_000_000,  # Hướng 1: TIER2 real-like, large scale
}

ALL_SIZE_LABELS: list[str] = list({
    **BENCHMARK_SIZES,
    **SCALABILITY_SIZES,
    **{k: 0 for k in GB_SIZES},
    **{k: 0 for k in SYNTHETIC_STRESS_SIZES},
}.keys())


# ─────────────────────────────────────────────────────────
# Dataset Schema — Amazon Reviews
# ─────────────────────────────────────────────────────────

SCHEMA_COLUMNS = [
    "review_id",
    "user_id",
    "product_id",
    "parent_asin",
    "rating",
    "review_title",
    "review_text",
    "review_time",
    "helpful_vote",
    "verified_purchase",
]

AMAZON_FIELD_MAP = {
    "asin": "product_id",
    "parent_asin": "parent_asin",
    "title": "review_title",
    "text": "review_text",
    "timestamp": "review_time",
    "helpful_vote": "helpful_vote",
    "verified_purchase": "verified_purchase",

    # Legacy UCSD format
    "reviewerID": "user_id",
    "reviewText": "review_text",
    "overall": "rating",
    "unixReviewTime": "review_time",
    "summary": "review_title",
    "verified": "verified_purchase",
}


# ─────────────────────────────────────────────────────────
# Calibrated Real-Data Parameters
# ─────────────────────────────────────────────────────────

CALIBRATED_FROM = "real/1M"
CALIBRATED_ROWS = 1_000_000

CALIBRATED_UNIQUE_USERS = 119_926
CALIBRATED_UNIQUE_PRODUCT_IDS = 771_319
CALIBRATED_UNIQUE_PARENT_ASINS = 476_112

ITEM_KEY_COLUMN = "parent_asin"

N_USERS = CALIBRATED_UNIQUE_USERS
N_PRODUCTS = CALIBRATED_UNIQUE_PARENT_ASINS
N_PRODUCT_IDS = CALIBRATED_UNIQUE_PRODUCT_IDS

# Text length calibration.
# The realistic generator uses log-normal text lengths. p85 remains available
# for memory-constrained tiers; p99 is used by tier2 to preserve the real tail.
TEXT_MIN_LEN = 12     # p5  of real/1M — loại bỏ review quá ngắn
TEXT_MAX_LEN = 374    # p85 of real/1M — memory-constrained cap
TEXT_GENERATOR_MAX_LEN = 1_234  # p99 of real/1M — realistic validation cap

TEXT_VARIANTS = 2_000
RANDOM_INSERT_PROB = 0.02

TEXT_LEN_LOGNORMAL_MU = 4.7013
TEXT_LEN_LOGNORMAL_SIGMA = 1.1979
TEXT_LEN_P50 = 122
TEXT_LEN_P25 = 51
TEXT_LEN_P75 = 259
TEXT_LEN_P85 = 374
TEXT_LEN_P90 = 472
TEXT_LEN_P95 = 654
TEXT_LEN_P99 = 1_234
TEXT_LEN_QUANTILE_POINTS = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
TEXT_LEN_QUANTILE_VALUES = [12, 51, 122, 259, 472, 654, 1_234]

ID_POOL_REUSE_PROB = 0.8801
PARENT_ASIN_REUSE_PROB = 0.5239
USER_SINGLE_REVIEW_PCT = 0.2183

USER_ZIPF_ALPHA = 0.5500
PARENT_ASIN_ZIPF_ALPHA = 0.7012

REAL_MEMORY_BYTES_PER_ROW = 683.9

RATING_DISTRIBUTION = [0.0579, 0.0504, 0.0909, 0.1629, 0.6379]

# helpful_vote is zero-heavy but has a right tail.
# Target reference from real: p90≈2, p95≈4, p99≈16.
HELPFUL_ZERO_PROB = 0.75
HELPFUL_LOGNORMAL_MEAN = 0.70
HELPFUL_LOGNORMAL_SIGMA = 1.15
HELPFUL_MAX = 1_000


# ─────────────────────────────────────────────────────────
# Benchmark Run Settings
# ─────────────────────────────────────────────────────────

WARMUP_RUNS = 1
BENCHMARK_RUNS = 3
TIMEOUT_SECONDS = 600


# ─────────────────────────────────────────────────────────
# Workload Parameters
# ─────────────────────────────────────────────────────────

WORKLOADS = ["filter", "groupby", "join", "pipeline"]

FILTER_RATING_THRESHOLD = 4

# Product-level benchmark key.
GROUPBY_COLUMN = ITEM_KEY_COLUMN
JOIN_KEY_COLUMN = ITEM_KEY_COLUMN

PRODUCT_METADATA_PATH = PROCESSED_DIR / "product_metadata.parquet"


# ─────────────────────────────────────────────────────────
# Framework Settings
# ─────────────────────────────────────────────────────────

FRAMEWORKS = ["pandas", "polars", "dask"]

PANDAS_CHUNK_SIZE = 100_000

POLARS_STREAMING = True
POLARS_N_THREADS = None

DASK_PARTITION_SIZE = "128MB"
DASK_N_WORKERS = 2
DASK_THREADS_PER_WORKER = 2
DASK_MEMORY_LIMIT = "4GB"


# ─────────────────────────────────────────────────────────
# IO Settings
# ─────────────────────────────────────────────────────────

DEFAULT_FORMAT = "parquet"
PARQUET_COMPRESSION = "snappy"
GENERATOR_CHUNK = 2_000_000


# ─────────────────────────────────────────────────────────
# Memory Profiler
# ─────────────────────────────────────────────────────────

MEMORY_POLL_INTERVAL = 0.05


# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_FILE = LOG_DIR / "benchmark.log"