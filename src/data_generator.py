"""
data_generator.py
Generates synthetic Amazon-like review datasets at configurable sizes.
Also supports loading real Amazon review data (CSV/JSON format).

Synthetic generation uses only numpy + pandas (no Faker dependency)
for speed at 100M-row scale.
"""

import gc
import hashlib
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from config import (
    CATEGORIES,
    COLUMNS,
    N_PRODUCTS,
    N_USERS,
    RATING_DISTRIBUTION,
    SYNTHETIC_DIR,
    PROCESSED_DIR,
    DATASET_SIZES,
)
from utils import get_logger, get_file_size_mb

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Review text templates
# ─────────────────────────────────────────────
_TEMPLATES = [
    "This product is absolutely {adj}. {detail}",
    "I {verb} this item. {detail}",
    "After {days} days of use, I can say it is {adj}. {detail}",
    "{adj} quality for the price. {detail}",
    "Would {rec} to anyone looking for this type of product. {detail}",
]
_ADJECTIVES = [
    "amazing", "terrible", "decent", "outstanding", "mediocre",
    "excellent", "poor", "fantastic", "average", "superb",
]
_DETAILS = [
    "Works as advertised.",
    "Fast shipping too.",
    "Not what I expected.",
    "Great customer service.",
    "Would buy again.",
    "Packaging was damaged.",
    "Easy to set up.",
    "Durable and well-made.",
    "Stopped working after a week.",
    "Exceeded my expectations.",
]
_VERBS = ["love", "hate", "enjoy", "recommend", "regret buying"]
_RECS = ["recommend", "not recommend"]


def _random_text(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate n short review text strings quickly using numpy choice."""
    templates = rng.choice(_TEMPLATES, n)
    adjs = rng.choice(_ADJECTIVES, n)
    details = rng.choice(_DETAILS, n)
    verbs = rng.choice(_VERBS, n)
    recs = rng.choice(_RECS, n)
    days = rng.integers(1, 365, n).astype(str)

    texts = []
    for t, adj, det, verb, rec, day in zip(templates, adjs, details, verbs, recs, days):
        txt = (
            t.replace("{adj}", adj)
             .replace("{detail}", det)
             .replace("{verb}", verb)
             .replace("{rec}", rec)
             .replace("{days}", day)
        )
        texts.append(txt)
    return np.array(texts, dtype=object)


# ─────────────────────────────────────────────
# Core Generator
# ─────────────────────────────────────────────

class SyntheticReviewGenerator:
    """
    Generates a synthetic Amazon Reviews DataFrame in chunks to avoid
    OOM when creating very large datasets (50M–100M rows).

    Args:
        n_rows: Total number of review rows to generate.
        seed: Random seed for reproducibility.
        chunk_size: Rows generated per chunk (controls peak RAM during generation).
    """

    CHUNK_SIZE = 2_000_000  # 2M rows per chunk (~300 MB peak)

    def __init__(self, n_rows: int, seed: int = 42, chunk_size: int = CHUNK_SIZE):
        self.n_rows = n_rows
        self.seed = seed
        self.chunk_size = chunk_size
        self.rng = np.random.default_rng(seed)

    def _generate_chunk(self, chunk_rows: int, offset: int) -> pd.DataFrame:
        rng = self.rng

        # IDs
        review_ids = [
            f"R{hashlib.md5(f'{offset+i}'.encode()).hexdigest()[:10].upper()}"
            for i in range(chunk_rows)
        ]
        user_ids = [f"U{x:08d}" for x in rng.integers(0, N_USERS, chunk_rows)]
        product_ids = [f"P{x:08d}" for x in rng.integers(0, N_PRODUCTS, chunk_rows)]

        # Rating (skewed positive distribution)
        ratings = rng.choice([1, 2, 3, 4, 5], size=chunk_rows, p=RATING_DISTRIBUTION)

        # Review text
        review_texts = _random_text(rng, chunk_rows)

        # Dates: random timestamps between 2010-01-01 and 2024-12-31
        start_ts = pd.Timestamp("2010-01-01").timestamp()
        end_ts = pd.Timestamp("2024-12-31").timestamp()
        timestamps = rng.uniform(start_ts, end_ts, chunk_rows)
        review_times = pd.to_datetime(timestamps, unit="s").normalize()

        # Category
        categories = rng.choice(CATEGORIES, chunk_rows)

        # Verified purchase (80% True)
        verified = rng.random(chunk_rows) < 0.80

        return pd.DataFrame({
            "review_id": review_ids,
            "user_id": user_ids,
            "product_id": product_ids,
            "rating": ratings.astype(np.int8),
            "review_text": review_texts,
            "review_time": review_times,
            "category": pd.Categorical(categories, categories=CATEGORIES),
            "verified_purchase": verified,
        })

    def generate(self) -> pd.DataFrame:
        """Generate and return the full dataset (use for small/medium sizes)."""
        logger.info(f"Generating {self.n_rows:,} rows in memory …")
        chunks = []
        offset = 0
        remaining = self.n_rows

        while remaining > 0:
            size = min(self.chunk_size, remaining)
            chunks.append(self._generate_chunk(size, offset))
            offset += size
            remaining -= size
            logger.debug(f"  chunk done — {offset:,}/{self.n_rows:,}")

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Generation complete: {len(df):,} rows, "
                    f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        return df

    def generate_to_parquet(self, path: Path, compression: str = "snappy") -> Path:
        """
        Stream-generate and write directly to Parquet in chunks.
        This keeps peak RAM low even for huge datasets.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Streaming {self.n_rows:,} rows → {path}")

        writer = None
        offset = 0
        remaining = self.n_rows
        t0 = time.perf_counter()

        try:
            while remaining > 0:
                size = min(self.chunk_size, remaining)
                chunk = self._generate_chunk(size, offset)
                table = pa.Table.from_pandas(chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        path, table.schema, compression=compression
                    )
                writer.write_table(table)

                offset += size
                remaining -= size
                elapsed = time.perf_counter() - t0
                pct = offset / self.n_rows * 100
                logger.info(f"  {pct:.0f}% — {offset:,}/{self.n_rows:,} rows "
                            f"({elapsed:.1f}s elapsed)")
                del chunk, table
                gc.collect()
        finally:
            if writer:
                writer.close()

        file_mb = get_file_size_mb(path)
        logger.info(f"Saved: {path.name} ({file_mb:.1f} MB)")
        return path

    def generate_to_csv(self, path: Path) -> Path:
        """Stream-generate and write directly to CSV (slower, larger files)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Streaming {self.n_rows:,} rows → {path} (CSV)")

        offset = 0
        remaining = self.n_rows
        header = True

        while remaining > 0:
            size = min(self.chunk_size, remaining)
            chunk = self._generate_chunk(size, offset)
            chunk.to_csv(path, mode="a", index=False, header=header)
            header = False
            offset += size
            remaining -= size
            logger.debug(f"  {offset:,}/{self.n_rows:,}")
            del chunk
            gc.collect()

        file_mb = get_file_size_mb(path)
        logger.info(f"Saved: {path.name} ({file_mb:.1f} MB)")
        return path


# ─────────────────────────────────────────────
# Product Metadata Table (for JOIN workload)
# ─────────────────────────────────────────────

def generate_product_metadata(path: Path = None) -> pd.DataFrame:
    """
    Generate a small product metadata table for JOIN experiments.
    Columns: product_id, category, price, brand, avg_rating_global
    """
    if path is None:
        path = PROCESSED_DIR / "product_metadata.parquet"

    if path.exists():
        logger.info(f"Product metadata already exists: {path}")
        return pd.read_parquet(path)

    rng = np.random.default_rng(999)
    product_ids = [f"P{x:08d}" for x in range(N_PRODUCTS)]

    brands = [f"Brand_{i}" for i in range(5000)]

    df = pd.DataFrame({
        "product_id": product_ids,
        "category": pd.Categorical(
            rng.choice(CATEGORIES, N_PRODUCTS), categories=CATEGORIES
        ),
        "price": np.round(rng.uniform(5.0, 999.0, N_PRODUCTS), 2),
        "brand": rng.choice(brands, N_PRODUCTS),
        "avg_rating_global": np.round(rng.uniform(1.0, 5.0, N_PRODUCTS), 2),
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Product metadata saved: {path} ({len(df):,} rows)")
    return df


# ─────────────────────────────────────────────
# High-level dataset preparation API
# ─────────────────────────────────────────────

def prepare_dataset(
    size_label: str,
    fmt: Literal["parquet", "csv"] = "parquet",
    force: bool = False,
) -> Path:
    """
    Ensure the dataset for a given size label exists on disk.
    Generates it if missing (or force=True).

    Args:
        size_label: One of the keys in DATASET_SIZES ("small", "medium", …)
        fmt: File format ("parquet" recommended for performance)
        force: Re-generate even if file exists

    Returns:
        Path to the dataset file.
    """
    n_rows = DATASET_SIZES[size_label]
    path = SYNTHETIC_DIR / f"reviews_{size_label}.{fmt}"

    if path.exists() and not force:
        size_mb = get_file_size_mb(path)
        logger.info(f"Dataset '{size_label}' already exists ({size_mb:.1f} MB): {path}")
        return path

    logger.info(f"Preparing dataset '{size_label}' ({n_rows:,} rows, format={fmt}) …")
    gen = SyntheticReviewGenerator(n_rows=n_rows)

    if fmt == "parquet":
        gen.generate_to_parquet(path)
    else:
        gen.generate_to_csv(path)

    # Also generate product metadata if not present
    generate_product_metadata()

    return path


def prepare_all_datasets(
    size_labels: list[str] = None,
    fmt: str = "parquet",
    force: bool = False,
) -> dict[str, Path]:
    """Prepare all datasets in the given list (defaults to all DATASET_SIZES)."""
    if size_labels is None:
        size_labels = list(DATASET_SIZES.keys())

    paths = {}
    for label in size_labels:
        try:
            paths[label] = prepare_dataset(label, fmt=fmt, force=force)
        except Exception as e:
            logger.error(f"Failed to prepare '{label}': {e}")
    return paths


# ─────────────────────────────────────────────
# Real Data Loader
# ─────────────────────────────────────────────

def load_real_amazon_data(
    source_path: Path,
    max_rows: int = None,
    output_path: Path = None,
) -> Path:
    """
    Load real Amazon review data (JSON-lines or CSV format)
    and standardize it to match the synthetic schema.

    Amazon review JSON fields expected:
        reviewerID, asin, reviewText, overall, unixReviewTime, verified

    Args:
        source_path: Path to the raw Amazon data file.
        max_rows: Optional row limit.
        output_path: Where to save the processed parquet. Defaults to processed/.

    Returns:
        Path to the standardized parquet file.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "real_amazon_reviews.parquet"

    if output_path.exists():
        logger.info(f"Real data already processed: {output_path}")
        return output_path

    logger.info(f"Loading real Amazon data from: {source_path}")
    suffix = source_path.suffix.lower()

    if suffix == ".json" or suffix == ".gz":
        # JSON-lines format (Amazon's standard distribution)
        chunks = []
        with pd.read_json(source_path, lines=True, chunksize=100_000) as reader:
            for chunk in reader:
                chunks.append(chunk)
                if max_rows and sum(len(c) for c in chunks) >= max_rows:
                    break
        df = pd.concat(chunks, ignore_index=True)
    elif suffix == ".csv":
        df = pd.read_csv(source_path, nrows=max_rows)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Expected .json, .gz, or .csv")

    if max_rows:
        df = df.head(max_rows)

    # Rename to match schema
    rename_map = {
        "reviewerID": "user_id",
        "asin": "product_id",
        "reviewText": "review_text",
        "overall": "rating",
        "unixReviewTime": "review_time",
        "verified": "verified_purchase",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Synthetic review_id
    df["review_id"] = [f"R{i:010d}" for i in range(len(df))]

    # Convert review_time
    if "review_time" in df.columns and df["review_time"].dtype in (int, float):
        df["review_time"] = pd.to_datetime(df["review_time"], unit="s").dt.normalize()

    # Add category if missing
    if "category" not in df.columns:
        df["category"] = "Unknown"

    # Keep only schema columns that exist
    keep_cols = [c for c in COLUMNS if c in df.columns]
    df = df[keep_cols]

    df.to_parquet(output_path, index=False)
    logger.info(f"Real data processed: {len(df):,} rows → {output_path}")
    return output_path


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark datasets")
    parser.add_argument(
        "--sizes", nargs="+",
        choices=list(DATASET_SIZES.keys()) + ["all"],
        default=["small", "medium"],
        help="Dataset sizes to generate",
    )
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--force", action="store_true", help="Regenerate if exists")
    args = parser.parse_args()

    sizes = list(DATASET_SIZES.keys()) if "all" in args.sizes else args.sizes
    prepare_all_datasets(sizes, fmt=args.format, force=args.force)
