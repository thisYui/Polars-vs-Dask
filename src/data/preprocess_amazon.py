"""
src/data/preprocess_amazon.py
Read raw Amazon .jsonl.gz files → clean → save as single Parquet in data/processed/.

Pipeline:
    1. Read all .gz files in data/raw/
    2. Rename fields to internal schema
    3. Cast & validate types
    4. Drop nulls in key columns
    5. Deduplicate on review_id
    6. Save → data/processed/amazon_reviews.parquet

Usage:
    python src/data/preprocess_amazon.py
    python src/data/preprocess_amazon.py --input data/raw --output data/processed/amazon_reviews.parquet
    python src/data/preprocess_amazon.py --max-rows 5000000
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.config import (
    RAW_DIR, PROCESSED_DIR,
    AMAZON_FIELD_MAP, SCHEMA_COLUMNS,
    CATEGORIES, N_PRODUCTS,
)
from src.utils import get_logger, get_file_size_mb

logger = get_logger("data.preprocess")

OUTPUT_PATH    = PROCESSED_DIR / "amazon_reviews.parquet"
CHUNK_SIZE     = 200_000   # rows per read chunk


# ─────────────────────────────────────────────────────────
# Schema casting
# ─────────────────────────────────────────────────────────

def _cast_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, cast, and validate one chunk."""
    # Rename fields
    df = df.rename(columns={k: v for k, v in AMAZON_FIELD_MAP.items() if k in df.columns})

    # review_id: synthesize if missing (asin + reviewerID hash)
    if "review_id" not in df.columns:
        uid = (df.get("user_id", "").astype(str) + df.get("product_id", "").astype(str))
        df["review_id"] = "R" + uid.apply(lambda x: str(abs(hash(x)))[:10])

    # rating → int8, clamp to 1–5
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].between(1, 5)]
        df["rating"] = df["rating"].astype("int8")

    # review_time → datetime (unix timestamp or string)
    if "review_time" in df.columns:
        col = df["review_time"]
        if pd.api.types.is_numeric_dtype(col):
            df["review_time"] = pd.to_datetime(col, unit="s", errors="coerce").dt.normalize()
        else:
            df["review_time"] = pd.to_datetime(col, errors="coerce").dt.normalize()

    # verified_purchase → bool
    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].astype(bool)

    # review_text → string, fill NaN
    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].fillna("").astype(str)

    # category → from 'category' column in Amazon data (list → first element)
    if "category" not in df.columns:
        df["category"] = "Unknown"
    else:
        # Amazon stores category as a list; take first element
        def _first_cat(val):
            if isinstance(val, list):
                return val[0] if val else "Unknown"
            return str(val) if pd.notna(val) else "Unknown"
        df["category"] = df["category"].apply(_first_cat)

    # Drop rows missing critical columns
    critical = ["review_id", "user_id", "product_id", "rating"]
    for col in critical:
        if col in df.columns:
            df = df[df[col].notna() & (df[col] != "")]

    # Keep only schema columns that exist
    keep = [c for c in SCHEMA_COLUMNS if c in df.columns]
    return df[keep].reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# Main preprocess function
# ─────────────────────────────────────────────────────────

def preprocess(
    raw_dir:    Path = RAW_DIR,
    output:     Path = OUTPUT_PATH,
    max_rows:   int  = None,
) -> Path:
    """
    Read all .jsonl.gz files in raw_dir, process, and write one Parquet file.

    Args:
        raw_dir  : directory containing .jsonl.gz files
        output   : destination parquet path
        max_rows : optional row cap (useful for testing)

    Returns:
        Path to the output parquet file.
    """
    gz_files = sorted(raw_dir.glob("*.jsonl.gz")) + sorted(raw_dir.glob("*.json.gz"))

    if not gz_files:
        logger.error(f"No .jsonl.gz files found in {raw_dir}")
        logger.info("Run: python src/data/download_amazon.py --small")
        raise FileNotFoundError(f"No gz files in {raw_dir}")

    logger.info(f"Found {len(gz_files)} raw files:")
    for f in gz_files:
        logger.info(f"  {f.name} ({get_file_size_mb(f):.1f} MB)")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer      = None
    total_rows  = 0
    total_dupes = 0
    seen_ids    = set()

    for gz_path in gz_files:
        logger.info(f"\nProcessing: {gz_path.name}")
        file_rows = 0

        try:
            reader = pd.read_json(
                gz_path,
                lines=True,
                chunksize=CHUNK_SIZE,
                compression="gzip",
            )

            for chunk in reader:
                chunk = _cast_chunk(chunk)

                # Deduplicate within chunk + against seen
                if "review_id" in chunk.columns:
                    before = len(chunk)
                    chunk = chunk[~chunk["review_id"].isin(seen_ids)]
                    chunk = chunk.drop_duplicates(subset=["review_id"])
                    dupes = before - len(chunk)
                    total_dupes += dupes
                    seen_ids.update(chunk["review_id"].tolist())

                if chunk.empty:
                    continue

                # Optional row cap
                if max_rows and total_rows + len(chunk) > max_rows:
                    chunk = chunk.iloc[: max_rows - total_rows]

                # Write to parquet
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="snappy")
                writer.write_table(table)

                total_rows += len(chunk)
                file_rows  += len(chunk)
                logger.debug(f"  chunk {len(chunk):,} rows | total {total_rows:,}")

                del chunk, table
                gc.collect()

                if max_rows and total_rows >= max_rows:
                    logger.info(f"  Reached max_rows={max_rows:,}, stopping.")
                    break

        except Exception as exc:
            logger.error(f"  Error processing {gz_path.name}: {exc}")
            continue

        logger.info(f"  {gz_path.name}: {file_rows:,} rows added")
        if max_rows and total_rows >= max_rows:
            break

    if writer:
        writer.close()

    size_mb = get_file_size_mb(output)
    logger.info(f"\n{'='*55}")
    logger.info(f"  Preprocessing complete")
    logger.info(f"  Total rows     : {total_rows:,}")
    logger.info(f"  Duplicates     : {total_dupes:,}")
    logger.info(f"  Output         : {output}")
    logger.info(f"  Output size    : {size_mb:.1f} MB")
    logger.info(f"{'='*55}")

    return output


def generate_product_metadata(processed_parquet: Path = OUTPUT_PATH) -> Path:
    """
    Extract unique products from the processed reviews and save a
    product metadata table for the JOIN workload.
    """
    from src.core.config import PRODUCT_METADATA_PATH

    if PRODUCT_METADATA_PATH.exists():
        logger.info(f"Product metadata already exists: {PRODUCT_METADATA_PATH}")
        return PRODUCT_METADATA_PATH

    logger.info("Extracting product metadata from processed reviews …")
    df = pd.read_parquet(
        processed_parquet,
        columns=["product_id", "category", "rating"],
    )
    meta = (
        df.groupby("product_id")
        .agg(
            category=("category", "first"),
            avg_rating_global=("rating", "mean"),
            review_count=("rating", "count"),
        )
        .reset_index()
    )
    meta["avg_rating_global"] = meta["avg_rating_global"].round(2)

    # Add synthetic price & brand (not in Amazon data)
    import numpy as np
    rng = np.random.default_rng(42)
    meta["price"] = rng.uniform(5.0, 999.0, len(meta)).round(2)
    meta["brand"] = [f"Brand_{i % 5000}" for i in range(len(meta))]

    PRODUCT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(PRODUCT_METADATA_PATH, index=False)
    logger.info(f"Saved product metadata: {PRODUCT_METADATA_PATH} ({len(meta):,} products)")
    return PRODUCT_METADATA_PATH


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw Amazon review files")
    parser.add_argument("--input",    default=str(RAW_DIR),     help="Raw data directory")
    parser.add_argument("--output",   default=str(OUTPUT_PATH), help="Output parquet path")
    parser.add_argument("--max-rows", type=int, default=None,   help="Row cap (for testing)")
    parser.add_argument("--metadata", action="store_true",       help="Also extract product metadata")
    args = parser.parse_args()

    out = preprocess(
        raw_dir=Path(args.input),
        output=Path(args.output),
        max_rows=args.max_rows,
    )

    if args.metadata:
        generate_product_metadata(out)