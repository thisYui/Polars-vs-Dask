"""
src/data/data_generator.py
Fast synthetic Amazon-like review generator.
Uses numpy only (no Faker) for speed at 100 M-row scale.
Streams data to Parquet in chunks to keep peak RAM low.

Usage:
    python src/data/data_generator.py --sizes 1M 10M 100M
    python src/data/data_generator.py --sizes 100M --format csv
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.config import (
    CATEGORIES, N_PRODUCTS, N_USERS,
    RATING_DISTRIBUTION,
    SYNTHETIC_DIR, PROCESSED_DIR,
    GENERATOR_CHUNK, PARQUET_COMPRESSION,
    BENCHMARK_SIZES, SCALABILITY_SIZES,
)
from src.utils import get_logger, get_file_size_mb

logger = get_logger("data.generator")

# ─────────────────────────────────────────────────────────
# Review text pool (numpy-based, no Faker dependency)
# ─────────────────────────────────────────────────────────
_TEMPLATES = [
    "This product is absolutely {adj}. {detail}",
    "I {verb} this item. {detail}",
    "After {days} days of use, I can say it is {adj}. {detail}",
    "{adj} quality for the price. {detail}",
    "Would {rec} to anyone. {detail}",
]
_ADJ   = ["amazing", "terrible", "decent", "outstanding", "mediocre",
           "excellent", "poor", "fantastic", "average", "superb"]
_DETAIL= ["Works as advertised.", "Fast shipping too.", "Not what I expected.",
          "Great customer service.", "Would buy again.", "Packaging was damaged.",
          "Easy to set up.", "Durable and well-made.",
          "Stopped working after a week.", "Exceeded my expectations."]
_VERB  = ["love", "hate", "enjoy", "recommend", "regret buying"]
_REC   = ["recommend", "not recommend"]


def _make_texts(rng: np.random.Generator, n: int) -> list[str]:
    tmpl  = rng.choice(_TEMPLATES, n)
    adjs  = rng.choice(_ADJ, n)
    dets  = rng.choice(_DETAIL, n)
    verbs = rng.choice(_VERB, n)
    recs  = rng.choice(_REC, n)
    days  = rng.integers(1, 365, n).astype(str)

    return [
        t.replace("{adj}", a).replace("{detail}", d)
         .replace("{verb}", v).replace("{rec}", r).replace("{days}", dy)
        for t, a, d, v, r, dy in zip(tmpl, adjs, dets, verbs, recs, days)
    ]


# ─────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────

class SyntheticReviewGenerator:
    """
    Generates synthetic Amazon review data at configurable scale.

    Args:
        n_rows     : total rows to generate
        seed       : random seed
        chunk_size : rows per generation chunk (controls peak RAM)
    """

    def __init__(
        self,
        n_rows:     int,
        seed:       int = 42,
        chunk_size: int = GENERATOR_CHUNK,
    ):
        self.n_rows     = n_rows
        self.seed       = seed
        self.chunk_size = chunk_size

    def _generate_chunk(self, n: int, offset: int, rng: np.random.Generator) -> pd.DataFrame:
        user_ids    = [f"U{x:08d}" for x in rng.integers(0, N_USERS, n)]
        product_ids = [f"P{x:08d}" for x in rng.integers(0, N_PRODUCTS, n)]
        review_ids  = [f"R{(offset + i):012d}" for i in range(n)]
        ratings     = rng.choice([1, 2, 3, 4, 5], n, p=RATING_DISTRIBUTION).astype("int8")
        texts       = _make_texts(rng, n)

        start_ts = pd.Timestamp("2010-01-01").timestamp()
        end_ts   = pd.Timestamp("2024-12-31").timestamp()
        times    = pd.to_datetime(
            rng.uniform(start_ts, end_ts, n), unit="s"
        ).normalize()

        cats     = rng.choice(CATEGORIES, n)
        verified = (rng.random(n) < 0.80)

        return pd.DataFrame({
            "review_id":        review_ids,
            "user_id":          user_ids,
            "product_id":       product_ids,
            "rating":           ratings,
            "review_text":      texts,
            "review_time":      times,
            "category":         pd.Categorical(cats, categories=CATEGORIES),
            "verified_purchase": verified,
        })

    def generate(self) -> pd.DataFrame:
        """Generate full dataset in memory (use for small sizes only)."""
        rng    = np.random.default_rng(self.seed)
        chunks = []
        offset = 0
        rem    = self.n_rows
        while rem > 0:
            n = min(self.chunk_size, rem)
            chunks.append(self._generate_chunk(n, offset, rng))
            offset += n
            rem    -= n
        return pd.concat(chunks, ignore_index=True)

    def generate_to_parquet(
        self,
        path:        Path,
        compression: str = PARQUET_COMPRESSION,
    ) -> Path:
        """Stream-generate directly to Parquet. Low peak RAM even at 100 M rows."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rng    = np.random.default_rng(self.seed)
        writer = None
        offset = 0
        rem    = self.n_rows
        t0     = time.perf_counter()

        try:
            while rem > 0:
                n     = min(self.chunk_size, rem)
                chunk = self._generate_chunk(n, offset, rng)
                table = pa.Table.from_pandas(chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression=compression)
                writer.write_table(table)

                offset += n
                rem    -= n
                pct     = offset / self.n_rows * 100
                elapsed = time.perf_counter() - t0
                logger.info(f"  {pct:.0f}% — {offset:,}/{self.n_rows:,} rows | {elapsed:.1f}s")

                del chunk, table
                gc.collect()
        finally:
            if writer:
                writer.close()

        mb = get_file_size_mb(path)
        logger.info(f"Saved: {path.name} ({mb:.1f} MB)")
        return path

    def generate_to_csv(self, path: Path) -> Path:
        """Stream-generate to CSV (slower, larger files)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rng    = np.random.default_rng(self.seed)
        offset = 0
        rem    = self.n_rows
        header = True

        while rem > 0:
            n     = min(self.chunk_size, rem)
            chunk = self._generate_chunk(n, offset, rng)
            chunk.to_csv(path, mode="a", index=False, header=header)
            header  = False
            offset += n
            rem    -= n
            del chunk
            gc.collect()

        logger.info(f"Saved CSV: {path.name} ({get_file_size_mb(path):.1f} MB)")
        return path


# ─────────────────────────────────────────────────────────
# Product metadata (for JOIN workload)
# ─────────────────────────────────────────────────────────

def generate_product_metadata(
    path: Path = None,
    force: bool = False,
) -> pd.DataFrame:
    """Generate small product metadata table used by the JOIN workload."""
    from src.core.config import PRODUCT_METADATA_PATH
    if path is None:
        path = PRODUCT_METADATA_PATH

    if path.exists() and not force:
        return pd.read_parquet(path)

    rng        = np.random.default_rng(999)
    product_ids = [f"P{x:08d}" for x in range(N_PRODUCTS)]
    brands     = [f"Brand_{i}" for i in range(5000)]

    df = pd.DataFrame({
        "product_id":       product_ids,
        "category":         pd.Categorical(rng.choice(CATEGORIES, N_PRODUCTS), categories=CATEGORIES),
        "price":            rng.uniform(5.0, 999.0, N_PRODUCTS).round(2),
        "brand":            rng.choice(brands, N_PRODUCTS),
        "avg_rating_global": rng.uniform(1.0, 5.0, N_PRODUCTS).round(2),
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Product metadata: {path.name} ({len(df):,} rows)")
    return df


# ─────────────────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────────────────

def prepare_synthetic(
    size_label: str,
    fmt:        str  = "parquet",
    dest_dir:   Path = SYNTHETIC_DIR,
    force:      bool = False,
) -> Path:
    """Generate one synthetic dataset if it doesn't exist."""
    from src.core.config import SCALABILITY_SIZES
    all_sizes = {**BENCHMARK_SIZES, **SCALABILITY_SIZES}
    if size_label not in all_sizes:
        raise ValueError(f"Unknown size label '{size_label}'. Choose from {list(all_sizes)}")

    n_rows = all_sizes[size_label]
    path   = dest_dir / f"reviews_{size_label}.{fmt}"

    if path.exists() and not force:
        logger.info(f"Already exists: {path.name} ({get_file_size_mb(path):.1f} MB)")
        return path

    logger.info(f"Generating '{size_label}' ({n_rows:,} rows, {fmt}) …")
    gen = SyntheticReviewGenerator(n_rows=n_rows)
    if fmt == "parquet":
        gen.generate_to_parquet(path)
    else:
        gen.generate_to_csv(path)

    generate_product_metadata()
    return path


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.core.config import SCALABILITY_SIZES
    all_size_keys = list({**BENCHMARK_SIZES, **SCALABILITY_SIZES}.keys())

    parser = argparse.ArgumentParser(description="Generate synthetic benchmark datasets")
    parser.add_argument("--sizes", nargs="+", default=["1M", "10M"],
                        choices=all_size_keys, help="Size labels to generate")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--dest", default=str(SYNTHETIC_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for label in args.sizes:
        prepare_synthetic(label, fmt=args.format, dest_dir=Path(args.dest), force=args.force)