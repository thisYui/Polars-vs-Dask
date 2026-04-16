"""
src/data/split_dataset.py
Create benchmark-ready splits (1M / 10M / 100M rows) from either:
  - data/processed/amazon_partitioned/    (partitioned, preferred)
  - data/processed/amazon_reviews.parquet (single-file fallback)
  - data/synthetic/                       (synthetic fallback)

Splits are saved to data/benchmark/ as:
  - Single file:  data/benchmark/reviews_1M.parquet
  - Partitioned:  data/benchmark/1M/part-000.parquet  ...  (--partition flag)

Usage:
    python src/data/split_dataset.py
    python src/data/split_dataset.py --source real
    python src/data/split_dataset.py --source synthetic
    python src/data/split_dataset.py --sizes 1M 10M
    python src/data/split_dataset.py --sizes 10M --partition
    python src/data/split_dataset.py --force    # overwrite existing splits
"""

import argparse
import gc
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pyarrow.parquet as pq
import pyarrow as pa

from src.core.config import (
    PROCESSED_DIR, BENCHMARK_REAL_DIR,
    BENCHMARK_SYN_DIR, BENCHMARK_SIZES,
    SCALABILITY_SIZES,
)
from src.utils import get_logger, get_file_size_mb

logger = get_logger("data.split")

PROCESSED_PARQUET     = PROCESSED_DIR / "amazon_reviews.parquet"
PROCESSED_PARTITIONED = PROCESSED_DIR / "amazon_partitioned"
READ_CHUNK_SIZE       = 500_000   # rows per read chunk during splitting

# FIX Bug #1: target_file_mb mặc định giảm từ 512 → 128 MB
# 512 MB quá lớn với Amazon real data → chỉ tạo 3 files thay vì 10-20 files
# 128 MB cho ra ~10-20 files với 50M rows, phù hợp hơn cho parallel read
DEFAULT_TARGET_FILE_MB = 128


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _count_parquet_rows(path: Path) -> int:
    """
    Fast row count without loading data.
    Works for both single-file and partitioned directory.
    """
    if path.is_dir():
        total = 0
        for part in path.rglob("*.parquet"):
            total += pq.ParquetFile(part).metadata.num_rows
        return total
    return pq.ParquetFile(path).metadata.num_rows


def _iter_batches(source: Path, batch_size: int):
    """
    Yield PyArrow RecordBatches from a single-file or partitioned parquet source.
    Normalises both cases so callers don't need to care.
    """
    if source.is_dir():
        parts = sorted(source.rglob("*.parquet"))
        for part in parts:
            pf = pq.ParquetFile(part)
            for batch in pf.iter_batches(batch_size=batch_size):
                yield batch
    else:
        pf = pq.ParquetFile(source)
        for batch in pf.iter_batches(batch_size=batch_size):
            yield batch


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*.parquet")) / 1024**2


# ─────────────────────────────────────────────────────────
# Single-file splits
# ─────────────────────────────────────────────────────────

def split_from_parquet(
    source:  Path,
    sizes:   dict[str, int],
    dest:    Path = BENCHMARK_REAL_DIR,
    force:   bool = False,
) -> dict[str, Path]:
    """
    Read `source` (single-file or partitioned dir) in chunks and write
    flat `reviews_<label>.parquet` splits.

    Args:
        source : parquet file or partitioned directory
        sizes  : {label: n_rows}  e.g. {"1M": 1_000_000}
        dest   : output directory
        force  : overwrite existing split files

    Returns:
        {label: Path} for successfully written splits
    """
    dest.mkdir(parents=True, exist_ok=True)

    total_source = _count_parquet_rows(source)
    src_label    = source.name if source.is_file() else str(source)
    logger.info(f"Source: {src_label} ({total_source:,} rows)")

    valid_sizes = {k: v for k, v in sizes.items() if v <= total_source}
    skipped     = [k for k, v in sizes.items() if v > total_source]
    if skipped:
        logger.warning(
            f"Source only has {total_source:,} rows — skipping splits: {skipped}\n"
            f"  Run data_generator.py to create larger synthetic splits."
        )

    results: dict[str, Path] = {}

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

    max_needed = max(to_generate.values())
    writers:  dict[str, pq.ParquetWriter] = {}
    written:  dict[str, int]              = {k: 0 for k in to_generate}
    schema    = None
    total_read = 0

    try:
        for batch in _iter_batches(source, READ_CHUNK_SIZE):
            if total_read >= max_needed:
                break

            table = pa.Table.from_batches([batch])

            if schema is None:
                schema = table.schema
                for label, n_rows in to_generate.items():
                    out_path = dest / f"reviews_{label}.parquet"
                    writers[label] = pq.ParquetWriter(out_path, schema, compression="snappy")

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
        for label, w in writers.items():
            w.close()
            out_path = dest / f"reviews_{label}.parquet"
            logger.warning(
                f"Split '{label}': only {written[label]:,}/{to_generate[label]:,} rows "
                f"(source exhausted). Saved partial."
            )
            results[label] = out_path

    return results


# ─────────────────────────────────────────────────────────
# Partitioned splits
# ─────────────────────────────────────────────────────────

def split_to_partitions(
    source:         Path,
    size_label:     str,
    n_rows:         int,
    dest:           Path = BENCHMARK_REAL_DIR,
    target_file_mb: int  = DEFAULT_TARGET_FILE_MB,  # FIX Bug #1: 128 thay vì 512
    force:          bool = False,
) -> Path:
    """
    Split source into a folder of smaller parquet files (~target_file_mb each).
    Engines read the folder directly:
        Dask   → dd.read_parquet("data/benchmark/10M/")
        Polars → pl.scan_parquet("data/benchmark/10M/*.parquet")
        Pandas → pd.read_parquet("data/benchmark/10M/")

    Args:
        source         : parquet file or partitioned dir to read from
        size_label     : label like "10M"
        n_rows         : target row count
        dest           : parent output directory
        target_file_mb : target size per output file (default: 128 MB)
        force          : overwrite existing

    Returns:
        Path to the output directory.
    """
    out_dir  = dest / size_label
    existing = list(out_dir.glob("part-*.parquet"))

    if existing and not force:
        total_mb = _dir_size_mb(out_dir)
        logger.info(
            f"Partition '{size_label}' already exists "
            f"({len(existing)} files, {total_mb:.0f} MB) — skip"
        )
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # FIX Bug #1+2: Ước tính bytes_per_row từ n_rows TARGET (không phải toàn bộ source)
    # Lấy mẫu từ 1 file nhỏ nhất trong source để có ước tính bytes/row chính xác hơn
    source_rows = _count_parquet_rows(source)
    if source.is_dir():
        source_bytes = sum(p.stat().st_size for p in source.rglob("*.parquet"))
    else:
        source_bytes = source.stat().st_size

    # bytes_per_row tính từ toàn bộ source (có thể bị skew nếu source >> n_rows cần)
    # → nhân thêm hệ số điều chỉnh dựa trên tỷ lệ n_rows / source_rows
    # Amazon real data: review text dài → bytes/row lớn hơn synthetic
    bytes_per_row_raw = source_bytes / max(source_rows, 1)

    # Thêm safety factor 1.2 để tránh partition quá lớn
    # (Amazon text sau compress Snappy vẫn ~150-300 bytes/row tùy category)
    bytes_per_row = bytes_per_row_raw * 1.2

    rows_per_partition = max(
        100_000,
        int((target_file_mb * 1024**2) / bytes_per_row),
    )
    n_partitions = math.ceil(n_rows / rows_per_partition)

    logger.info(
        f"Splitting '{size_label}': {n_rows:,} rows → "
        f"~{n_partitions} files × ~{rows_per_partition:,} rows "
        f"(~{target_file_mb} MB each | {bytes_per_row:.0f} bytes/row estimated)"
    )

    writer       = None
    schema       = None
    written      = 0
    part_idx     = 0
    part_written = 0

    try:
        for batch in _iter_batches(source, 200_000):
            if written >= n_rows:
                break

            table = pa.Table.from_batches([batch])
            if schema is None:
                schema = table.schema

            remaining = n_rows - written
            chunk     = table.slice(0, min(len(table), remaining))
            offset    = 0

            while offset < len(chunk):
                if writer is None:
                    part_path = out_dir / f"part-{part_idx:03d}.parquet"
                    writer    = pq.ParquetWriter(part_path, schema, compression="snappy")
                    part_written = 0

                space_left = rows_per_partition - part_written
                take       = min(space_left, len(chunk) - offset)

                writer.write_table(chunk.slice(offset, take))
                part_written += take
                written      += take
                offset       += take

                if part_written >= rows_per_partition:
                    writer.close()
                    part_path = out_dir / f"part-{part_idx:03d}.parquet"
                    size_mb   = part_path.stat().st_size / 1024**2
                    logger.info(f"  ✓ part-{part_idx:03d}.parquet | {part_written:,} rows | {size_mb:.0f} MB")
                    writer    = None
                    part_idx += 1

            del table, chunk
            gc.collect()

    finally:
        if writer:
            writer.close()
            part_path = out_dir / f"part-{part_idx:03d}.parquet"
            size_mb   = part_path.stat().st_size / 1024**2
            logger.info(f"  ✓ part-{part_idx:03d}.parquet | {part_written:,} rows | {size_mb:.0f} MB")

    total_mb = _dir_size_mb(out_dir)
    n_files  = len(list(out_dir.glob("part-*.parquet")))
    logger.info(f"\nDone: {out_dir} | {written:,} rows | {n_files} files | {total_mb:.0f} MB total")
    return out_dir


# ─────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────

def split_from_synthetic(
    sizes:     dict[str, int],
    dest:      Path = BENCHMARK_SYN_DIR,
    force:     bool = False,
    partition: bool = False,
) -> dict[str, Path]:
    """
    Generate splits directly from the synthetic generator.
    Used when real data is unavailable or source is too small.
    """
    from src.data.data_generator import SyntheticReviewGenerator, generate_product_metadata

    results = {}
    for label, n_rows in sizes.items():
        if partition:
            out_path = dest / label
            existing = list(out_path.glob("part-*.parquet")) if out_path.exists() else []
            if existing and not force:
                logger.info(f"  Partition '{label}' already exists — skip. Use --force to regenerate.")
                results[label] = out_path
                continue
            # Generate to a temp single file, then partition it
            tmp_path = dest / f"_tmp_{label}.parquet"
            logger.info(f"Generating synthetic '{label}': {n_rows:,} rows …")
            SyntheticReviewGenerator(n_rows=n_rows).generate_to_parquet(tmp_path)
            split_to_partitions(tmp_path, label, n_rows, dest=dest, force=force)
            tmp_path.unlink(missing_ok=True)
            results[label] = out_path
        else:
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


# ─────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────

def prepare_benchmark_splits(
    source:         str  = "auto",
    sizes:          dict[str, int] = None,
    force:          bool = False,
    partition:      bool = False,
    target_file_mb: int  = DEFAULT_TARGET_FILE_MB,  # FIX Bug #1: expose parameter
) -> dict[str, Path]:
    """
    High-level entry point used by run_pipeline.py.

    source="auto": uses processed real data (partitioned preferred, single-file
                   fallback), otherwise falls back to synthetic.

    Args:
        source         : "real" | "synthetic" | "auto"
        sizes          : {label: n_rows}
        force          : overwrite existing outputs
        partition      : produce multi-file partition folders instead of single files
        target_file_mb : target MB per partition file (default: 128)
    """
    # FIX Bug #2 (CLI): SCALABILITY_SIZES chứa đủ tất cả keys kể cả "50M"
    # Dùng SCALABILITY_SIZES làm lookup thay vì BENCHMARK_SIZES
    sizes = sizes or BENCHMARK_SIZES

    if source == "synthetic":
        return split_from_synthetic(sizes, force=force, partition=partition)

    # Determine best real-data source
    real_source = None
    if PROCESSED_PARTITIONED.exists() and any(PROCESSED_PARTITIONED.rglob("*.parquet")):
        real_source = PROCESSED_PARTITIONED
        logger.info(f"Using partitioned source: {real_source}")
    elif PROCESSED_PARQUET.exists():
        real_source = PROCESSED_PARQUET
        logger.info(f"Using single-file source: {real_source}")

    if source == "real" and real_source is None:
        raise FileNotFoundError(
            "No processed real data found. "
            "Run: python src/data/preprocess_amazon.py [--partition]"
        )

    if real_source is not None:
        try:
            if partition:
                # Build partitioned benchmark splits from real data
                results    = {}
                total_rows = _count_parquet_rows(real_source)
                logger.info(f"Real source total rows: {total_rows:,}")

                for label, n_rows in sizes.items():
                    if n_rows > total_rows:
                        logger.warning(
                            f"  '{label}' needs {n_rows:,} rows but source has "
                            f"{total_rows:,} — skipping"
                        )
                        continue
                    results[label] = split_to_partitions(
                        real_source, label, n_rows,
                        force=force,
                        target_file_mb=target_file_mb,
                    )
                missing = {k: v for k, v in sizes.items() if k not in results}
            else:
                results = split_from_parquet(real_source, sizes, force=force)
                missing = {k: v for k, v in sizes.items() if k not in results}

            if missing:
                logger.info(f"Falling back to synthetic for: {list(missing)}")
                results.update(split_from_synthetic(missing, dest=BENCHMARK_SYN_DIR, force=force, partition=partition))

            return results

        except Exception as exc:
            logger.error(f"Real data split failed: {exc}")
            # FIX Bug #3: Không silently fallback khi --source real
            # User phải tường minh biết việc fallback xảy ra
            if source == "real":
                logger.error(
                    "  --source real được chỉ định — KHÔNG fallback sang synthetic.\n"
                    "  Hãy kiểm tra lại data/processed/ hoặc chạy lại preprocess."
                )
                raise
            logger.error("  Falling back to synthetic — dữ liệu benchmark sẽ là SYNTHETIC!")

    logger.info("No processed real data found — using synthetic generator.")
    return split_from_synthetic(sizes, force=force, partition=partition)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # FIX Bug #2: --sizes choices phải dùng SCALABILITY_SIZES (có đủ 1M/5M/10M/50M/100M)
    # mà default cũng từ BENCHMARK_SIZES để không thay đổi hành vi cũ
    _ALL_SIZE_KEYS = list(SCALABILITY_SIZES.keys())

    parser = argparse.ArgumentParser(description="Create benchmark dataset splits")
    parser.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto",
                        help="Data source: real Amazon, synthetic, or auto-detect")
    parser.add_argument(
        "--sizes", nargs="+",
        default=list(BENCHMARK_SIZES.keys()),
        choices=_ALL_SIZE_KEYS,                   # FIX: đủ keys kể cả 50M/100M
        help="Split labels to create",
    )
    parser.add_argument("--scalability", action="store_true",
                        help="Create all scalability sizes (1M/5M/10M/50M/100M)")
    parser.add_argument("--partition", action="store_true",
                        help="Write multi-file partition folders (recommended for 20M+)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits")
    parser.add_argument(
        "--target-file-mb", type=int, default=DEFAULT_TARGET_FILE_MB,
        help=(
            f"Target MB per partition file (default: {DEFAULT_TARGET_FILE_MB}).\n"
            "Giảm xuống nếu số file partition quá ít (vd: --target-file-mb 64)."
        ),
    )
    args = parser.parse_args()

    # FIX Bug #2: lookup từ SCALABILITY_SIZES (chứa đủ tất cả labels)
    if args.scalability:
        size_map = dict(SCALABILITY_SIZES)
    else:
        missing_keys = [k for k in args.sizes if k not in SCALABILITY_SIZES]
        if missing_keys:
            logger.error(
                f"Size keys không có trong SCALABILITY_SIZES: {missing_keys}\n"
                f"Hãy kiểm tra src/core/config.py và thêm các keys này vào SCALABILITY_SIZES."
            )
            sys.exit(1)
        size_map = {k: SCALABILITY_SIZES[k] for k in args.sizes}

    results = prepare_benchmark_splits(
        source=args.source,
        sizes=size_map,
        force=args.force,
        partition=args.partition,
        target_file_mb=args.target_file_mb,
    )
    print(f"\nReady splits ({len(results)}):")
    for label, path in sorted(results.items()):
        if path.is_dir():
            parts   = list(path.glob("part-*.parquet"))
            size_mb = _dir_size_mb(path)
            print(f"  {label:<8} {size_mb:.1f} MB ({len(parts)} files)  ->  {path}")
        else:
            print(f"  {label:<8} {get_file_size_mb(path):.1f} MB  ->  {path}")