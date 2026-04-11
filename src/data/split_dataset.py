"""
src/data/split_dataset.py
Create benchmark-ready splits (1M / 10M / 100M rows) from either:
  - data/processed/amazon_reviews.parquet  (real data, preferred)
  - data/synthetic/                        (synthetic fallback)

Splits are saved to data/benchmark/ as parquet files.

Usage:
    python src/data/split_dataset.py
    python src/data/split_dataset.py --source real
    python src/data/split_dataset.py --source synthetic
    python src/data/split_dataset.py --sizes 1M 10M
    python src/data/split_dataset.py --force    # overwrite existing splits
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pyarrow.parquet as pq
import pyarrow as pa

from src.core.config import (
    PROCESSED_DIR, SYNTHETIC_DIR, BENCHMARK_DIR,
    BENCHMARK_SIZES, SCALABILITY_SIZES,
)
from src.utils import get_logger, get_file_size_mb

logger = get_logger("data.split")

PROCESSED_PARQUET = PROCESSED_DIR / "amazon_reviews.parquet"
READ_CHUNK_SIZE   = 500_000   # rows per read chunk during splitting


def _count_parquet_rows(path: Path) -> int:
    """Fast row count without loading data."""
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows


def split_from_parquet(
    source:  Path,
    sizes:   dict[str, int],
    dest:    Path = BENCHMARK_DIR,
    force:   bool = False,
) -> dict[str, Path]:
    """
    Read `source` parquet in chunks and write `sizes` splits.

    Args:
        source : large parquet file to split
        sizes  : {label: n_rows} e.g. {"1M": 1_000_000, "10M": 10_000_000}
        dest   : output directory
        force  : overwrite existing split files

    Returns:
        {label: Path} for successfully written splits
    """
    dest.mkdir(parents=True, exist_ok=True)

    total_source = _count_parquet_rows(source)
    logger.info(f"Source: {source.name} ({total_source:,} rows, {get_file_size_mb(source):.1f} MB)")

    # Filter to sizes that fit in source
    valid_sizes = {k: v for k, v in sizes.items() if v <= total_source}
    skipped     = [k for k, v in sizes.items() if v > total_source]
    if skipped:
        logger.warning(f"Source only has {total_source:,} rows — skipping splits: {skipped}")
        logger.warning("  Run data_generator.py to create larger synthetic splits.")

    results: dict[str, Path] = {}
    max_needed = max(valid_sizes.values()) if valid_sizes else 0

    # Stream source once, write multiple splits simultaneously
    writers: dict[str, pq.ParquetWriter] = {}
    written: dict[str, int] = {k: 0 for k in valid_sizes}
    schema_set = False
    schema     = None

    # Determine which splits need generating
    to_generate = {}
    for label, n_rows in valid_sizes.items():
        out_path = dest / f"reviews_{label}.parquet"
        if out_path.exists() and not force:
            logger.info(f"  Split '{label}' already exists ({get_file_size_mb(out_path):.1f} MB) — skip")
            results[label] = out_path
        else:
            to_generate[label] = n_rows

    if not to_generate:
        logger.info("All splits already exist. Use --force to regenerate.")
        return results

    logger.info(f"Generating splits: {list(to_generate.keys())}")

    pf     = pq.ParquetFile(source)
    total_read = 0

    try:
        for batch in pf.iter_batches(batch_size=READ_CHUNK_SIZE):
            if total_read >= max_needed:
                break

            table = pa.Table.from_batches([batch])
            if not schema_set:
                schema = table.schema
                for label, n_rows in to_generate.items():
                    out_path = dest / f"reviews_{label}.parquet"
                    writers[label] = pq.ParquetWriter(out_path, schema, compression="snappy")
                schema_set = True

            chunk_len = len(table)

            for label, n_rows in to_generate.items():
                still_need = n_rows - written[label]
                if still_need <= 0:
                    continue
                take = min(chunk_len, still_need)
                writers[label].write_table(table.slice(0, take))
                written[label] += take

                if written[label] >= n_rows:
                    writers[label].close()
                    out_path = dest / f"reviews_{label}.parquet"
                    size_mb  = get_file_size_mb(out_path)
                    logger.info(f"  ✓ Split '{label}': {written[label]:,} rows | {size_mb:.1f} MB → {out_path.name}")
                    results[label] = out_path
                    del writers[label]

            total_read += chunk_len
            del table
            gc.collect()

    finally:
        # Close any still-open writers (source ran out before target size)
        for label, w in writers.items():
            w.close()
            out_path = dest / f"reviews_{label}.parquet"
            logger.warning(
                f"Split '{label}': only {written[label]:,}/{to_generate[label]:,} rows "
                f"(source exhausted). Saved partial."
            )
            results[label] = out_path

    return results


def split_from_synthetic(
    sizes:  dict[str, int],
    dest:   Path = BENCHMARK_DIR,
    force:  bool = False,
) -> dict[str, Path]:
    """
    Generate splits directly from the synthetic generator.
    Used when real data is unavailable or source is too small.
    """
    from src.data.data_generator import SyntheticReviewGenerator, generate_product_metadata

    results = {}
    for label, n_rows in sizes.items():
        out_path = dest / f"reviews_{label}.parquet"
        if out_path.exists() and not force:
            logger.info(f"  Split '{label}' already exists — skip. Use --force to regenerate.")
            results[label] = out_path
            continue

        logger.info(f"Generating synthetic split '{label}': {n_rows:,} rows …")
        gen  = SyntheticReviewGenerator(n_rows=n_rows)
        path = gen.generate_to_parquet(out_path)
        results[label] = path
        gc.collect()

    generate_product_metadata()
    return results


def prepare_benchmark_splits(
    source: str = "auto",   # "real" | "synthetic" | "auto"
    sizes:  dict[str, int] = None,
    force:  bool = False,
) -> dict[str, Path]:
    """
    High-level entry point used by run_all.py.

    source="auto": uses real processed data if available, else synthetic.
    """
    sizes = sizes or BENCHMARK_SIZES

    if source == "synthetic":
        return split_from_synthetic(sizes, force=force)

    if source == "real" or (source == "auto" and PROCESSED_PARQUET.exists()):
        try:
            results = split_from_parquet(PROCESSED_PARQUET, sizes, force=force)
            # For sizes that could not be covered by real data, fall back to synthetic
            missing = {k: v for k, v in sizes.items() if k not in results}
            if missing:
                logger.info(f"Falling back to synthetic for: {list(missing)}")
                results.update(split_from_synthetic(missing, force=force))
            return results
        except Exception as exc:
            logger.error(f"Real data split failed: {exc} — falling back to synthetic")
            return split_from_synthetic(sizes, force=force)

    # auto + no real data
    logger.info("No processed real data found — using synthetic generator.")
    return split_from_synthetic(sizes, force=force)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create benchmark dataset splits")
    parser.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto",
                        help="Data source: real Amazon, synthetic, or auto-detect")
    parser.add_argument("--sizes", nargs="+", default=list(BENCHMARK_SIZES.keys()),
                        choices=list(SCALABILITY_SIZES.keys()),
                        help="Split labels to create")
    parser.add_argument("--scalability", action="store_true",
                        help="Create all scalability sizes (1M/5M/10M/50M/100M)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits")
    args = parser.parse_args()

    size_map = SCALABILITY_SIZES if args.scalability else {
        k: SCALABILITY_SIZES[k] for k in args.sizes
    }

    results = prepare_benchmark_splits(
        source=args.source,
        sizes=size_map,
        force=args.force,
    )
    print(f"\nReady splits ({len(results)}):")
    for label, path in sorted(results.items()):
        print(f"  {label:<8} {get_file_size_mb(path):.1f} MB  →  {path}")