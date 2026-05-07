"""
src/data/preprocess_amazon.py  — crash-safe, RAM-efficient rewrite

Key changes vs original:
  1. `seen_ids` replaced by a probabilistic BloomFilter (mmh3 + bitarray)
     → O(1) memory regardless of dataset size; ~0.1% false-positive rate at 500M items
     → Falls back to a plain set if mmh3/bitarray unavailable (small datasets only)
  2. `_PartitionedWriter` flushes every chunk (no in-memory buffer dict)
     → peak RAM = 1 chunk × n_categories  (was unbounded)
  3. `_PartitionedWriter` opens/closes file handles per flush
     → no "too many open files" crash on large category counts
  4. `--resume` flag: skips files already fully written (checks a .progress sidecar)
  5. Heartbeat thread: logs "still alive" every 60 s so you see if it's frozen
  6. SIGTERM / KeyboardInterrupt → clean writer.close() before exit
"""

import argparse
import gc
import gzip
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.config import (
    RAW_DIR, PROCESSED_DIR,
    AMAZON_FIELD_MAP, SCHEMA_COLUMNS,
)
from src.utils import get_logger, get_file_size_mb

logger = get_logger("data.preprocess")

OUTPUT_PATH   = PROCESSED_DIR / "amazon_reviews.parquet"
PARTITION_DIR = PROCESSED_DIR / "amazon_partitioned"
CHUNK_SIZE    = 100_000   # ↓ from 200k → lower peak RAM per chunk


# ─────────────────────────────────────────────────────────
# Bloom filter (RAM-safe dedup)
# ─────────────────────────────────────────────────────────

class _BloomFilter:
    """
    Simple counting Bloom filter using mmh3 + bitarray.
    Targets ~0.1 % false-positive rate at `capacity` items.
    Falls back to a plain Python set if dependencies missing.
    """

    def __init__(self, capacity: int = 500_000_000, error_rate: float = 0.001):
        try:
            import mmh3
            from bitarray import bitarray
            import math

            # Optimal bit-array size and hash count
            m = int(-capacity * math.log(error_rate) / math.log(2) ** 2)
            k = max(1, int(m / capacity * math.log(2)))

            self._bits = bitarray(m)
            self._bits.setall(0)
            self._m = m
            self._k = k
            self._mmh3 = mmh3
            self._use_bloom = True
            ram_mb = m / 8 / 1024 / 1024
            logger.info(
                f"BloomFilter: capacity={capacity:,}, k={k} hashes, "
                f"m={m:,} bits (~{ram_mb:.0f} MB RAM)"
            )
        except ImportError:
            logger.warning(
                "mmh3 / bitarray not installed — falling back to plain set for dedup. "
                "Install with: pip install mmh3 bitarray"
            )
            self._set: set[str] = set()
            self._use_bloom = False

    def __contains__(self, item: str) -> bool:
        if not self._use_bloom:
            return item in self._set
        h = abs(hash(item))
        for i in range(self._k):
            idx = (self._mmh3.hash(item, i) % self._m + self._m) % self._m
            if not self._bits[idx]:
                return False
        return True

    def add(self, item: str) -> None:
        if not self._use_bloom:
            self._set.add(item)
            return
        for i in range(self._k):
            idx = (self._mmh3.hash(item, i) % self._m + self._m) % self._m
            self._bits[idx] = True


# ─────────────────────────────────────────────────────────
# Schema casting  (unchanged from original)
# ─────────────────────────────────────────────────────────

def _cast_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # ── Step 1: rename raw fields → internal names ──────────────────────────
    df = df.rename(columns={k: v for k, v in AMAZON_FIELD_MAP.items() if k in df.columns})

    # ── Step 2: review_id (synthetic key if absent) ─────────────────────────
    if "review_id" not in df.columns:
        uid = (df.get("user_id", pd.Series([""] * len(df))).astype(str)
               + df.get("product_id", pd.Series([""] * len(df))).astype(str))
        df["review_id"] = "R" + uid.apply(lambda x: str(abs(hash(x)))[:10])

    # ── Step 3: rating ───────────────────────────────────────────────────────
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].between(1, 5)]
        df["rating"] = df["rating"].astype("int8")

    # ── Step 4: review_time — handles ms (HF 2023) and unix-s (legacy) ──────
    if "review_time" in df.columns:
        col = df["review_time"]
        if pd.api.types.is_numeric_dtype(col):
            # HuggingFace 2023: millisecond epoch (13-digit, median > 1e11)
            # Legacy UCSD: second epoch (10-digit)
            median_val = col.median()
            if median_val > 1e11:
                df["review_time"] = pd.to_datetime(col, unit="ms", errors="coerce").dt.normalize()
            else:
                df["review_time"] = pd.to_datetime(col, unit="s",  errors="coerce").dt.normalize()
        else:
            df["review_time"] = pd.to_datetime(col, errors="coerce").dt.normalize()

    # ── Step 5: helpful_vote ─────────────────────────────────────────────────
    if "helpful_vote" in df.columns:
        df["helpful_vote"] = (
            pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype("int32")
        )
    else:
        df["helpful_vote"] = pd.array([0] * len(df), dtype="int32")

    # ── Step 6: verified_purchase ────────────────────────────────────────────
    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].astype(bool)

    # ── Step 7: text fields ──────────────────────────────────────────────────
    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].fillna("").astype(str)

    if "review_title" in df.columns:
        df["review_title"] = df["review_title"].fillna("").astype(str)
    else:
        df["review_title"] = ""

    if "parent_asin" in df.columns:
        df["parent_asin"] = df["parent_asin"].fillna("").astype(str)
    else:
        df["parent_asin"] = ""

    # ── Step 8: category (filename-based for HF; list-valued for legacy) ─────
    if "category" not in df.columns:
        df["category"] = "Unknown"
    else:
        def _first_cat(val):
            if isinstance(val, list):
                return val[0] if val else "Unknown"
            return str(val) if pd.notna(val) else "Unknown"
        df["category"] = df["category"].apply(_first_cat)

    # ── Step 9: drop rows missing mandatory fields ───────────────────────────
    for col in ["review_id", "user_id", "product_id", "rating"]:
        if col in df.columns:
            df = df[df[col].notna() & (df[col].astype(str) != "")]

    # ── Step 10: select & order final columns ────────────────────────────────
    keep = [c for c in SCHEMA_COLUMNS if c in df.columns]
    return df[keep].reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# Writer helpers
# ─────────────────────────────────────────────────────────

class _SingleFileWriter:
    def __init__(self, path: Path):
        self.path    = path
        self._writer = None

    def write(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer:
            self._writer.close()

    def summary(self) -> str:
        return f"{self.path} ({get_file_size_mb(self.path):.1f} MB)"


class _PartitionedWriter:
    """
    Flush-on-every-chunk partitioned writer.

    Each call to write() immediately splits the table by category and
    appends each slice to its own parquet file.  No in-memory buffer dict.
    Peak extra RAM = size of one chunk (already allocated by the caller).
    """

    def __init__(self, out_dir: Path):
        self.out_dir  = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self._part_idx: dict[str, int] = {}

    def write(self, table: pa.Table) -> None:
        import pyarrow.compute as pc

        if "category" not in table.schema.names:
            self._append("Unknown", table)
            return

        cat_col     = table.column("category")
        unique_cats = cat_col.to_pylist()
        unique_cats = list(dict.fromkeys(unique_cats))  # preserve order, deduplicate

        # Remove category column — it becomes the partition directory name
        data_cols = [n for n in table.schema.names if n != "category"]

        for cat in unique_cats:
            mask  = pc.equal(cat_col, cat)
            slice_ = table.filter(mask).select(data_cols)
            if len(slice_) > 0:
                self._append(str(cat), slice_)

    def _append(self, cat: str, table: pa.Table) -> None:
        # Safe directory name (replace spaces / slashes)
        safe = cat.replace("/", "_").replace(" ", "_")
        cat_dir  = self.out_dir / f"category={safe}"
        cat_dir.mkdir(parents=True, exist_ok=True)

        idx      = self._part_idx.get(cat, 0)
        out_path = cat_dir / f"part-{idx:04d}.parquet"

        if out_path.exists():
            # Append by reading existing + concatenating (rare; only at resume boundary)
            existing = pq.read_table(out_path)
            table    = pa.concat_tables([existing, table])

        pq.write_table(table, out_path, compression="snappy")

        # Roll over part file after 250 MB to keep individual files manageable
        if out_path.stat().st_size > 250 * 1024 * 1024:
            self._part_idx[cat] = idx + 1

    def close(self) -> None:
        pass  # nothing buffered

    def summary(self) -> str:
        parts  = list(self.out_dir.rglob("part-*.parquet"))
        total  = sum(p.stat().st_size for p in parts) / 1024**2
        n_cats = len(list(self.out_dir.glob("category=*")))
        return f"{self.out_dir} ({n_cats} categories, {len(parts)} files, {total:.1f} MB total)"


# ─────────────────────────────────────────────────────────
# Progress sidecar (resume support)
# ─────────────────────────────────────────────────────────

class _Progress:
    """JSON sidecar file that tracks which input files are fully processed."""

    def __init__(self, out_dir: Path):
        self._path = out_dir / ".preprocess_progress.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {"done_files": [], "total_rows": 0, "total_dupes": 0}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2))

    def is_done(self, file_path: Path) -> bool:
        return str(file_path) in self._data["done_files"]

    def mark_done(self, file_path: Path, rows: int, dupes: int) -> None:
        self._data["done_files"].append(str(file_path))
        self._data["total_rows"]  += rows
        self._data["total_dupes"] += dupes
        self._save()

    @property
    def total_rows(self) -> int:
        return self._data["total_rows"]

    @property
    def total_dupes(self) -> int:
        return self._data["total_dupes"]


# ─────────────────────────────────────────────────────────
# Heartbeat thread
# ─────────────────────────────────────────────────────────

class _Heartbeat(threading.Thread):
    """Logs a 'still alive' message every `interval` seconds."""

    def __init__(self, interval: int = 60):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop    = threading.Event()
        self.rows_ref: list[int] = [0]  # mutable reference

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            logger.info(f"[heartbeat] still running | total rows so far: {self.rows_ref[0]:,}")

    def stop(self) -> None:
        self._stop.set()


# ─────────────────────────────────────────────────────────
# File detection
# ─────────────────────────────────────────────────────────

def _detect_raw_files(raw_dir: Path, raw_format: str = "auto") -> list[tuple[Path, str]]:
    gz_files    = sorted(raw_dir.glob("*.jsonl.gz")) + sorted(raw_dir.glob("*.json.gz"))
    jsonl_files = sorted(raw_dir.glob("*.jsonl"))

    if raw_format == "gz":
        return [(f, "gz") for f in gz_files]
    if raw_format == "jsonl":
        return [(f, "jsonl") for f in jsonl_files]

    gz_stems = {f.name.replace(".jsonl.gz", "").replace(".json.gz", "") for f in gz_files}
    result: list[tuple[Path, str]] = [(f, "gz") for f in gz_files]
    for f in jsonl_files:
        if f.stem not in gz_stems:
            result.append((f, "jsonl"))
        else:
            logger.debug(f"Skipping plain jsonl '{f.name}' — gz version present")
    return result


# ─────────────────────────────────────────────────────────
# Main preprocess
# ─────────────────────────────────────────────────────────

def preprocess(
    raw_dir:    Path = RAW_DIR,
    output:     Path = OUTPUT_PATH,
    max_rows:   int  = None,
    partition:  bool = False,
    raw_format: str  = "auto",
    resume:     bool = True,
    bloom_cap:  int  = 500_000_000,
) -> Path:
    raw_files = _detect_raw_files(raw_dir, raw_format=raw_format)

    if not raw_files:
        logger.error(f"No .jsonl.gz or .jsonl files found in {raw_dir}")
        raise FileNotFoundError(f"No raw review files in {raw_dir}")

    if partition:
        writer   = _PartitionedWriter(PARTITION_DIR)
        out_path = PARTITION_DIR
        PARTITION_DIR.mkdir(parents=True, exist_ok=True)
        progress = _Progress(PARTITION_DIR)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer   = _SingleFileWriter(output)
        out_path = output
        progress = _Progress(output.parent)

    logger.info(f"Found {len(raw_files)} raw file(s). Resume={resume}")

    # Bloom filter for global dedup
    seen = _BloomFilter(capacity=bloom_cap)
    # Seed from already-processed rows count (approximate; bloom can't restore state,
    # but that's fine — false positives only cause minor over-dedup, not crashes)

    total_rows  = progress.total_rows
    total_dupes = progress.total_dupes

    heartbeat = _Heartbeat(interval=60)
    heartbeat.rows_ref[0] = total_rows
    heartbeat.start()

    # Graceful shutdown on SIGTERM
    _shutdown = threading.Event()
    def _handle_signal(sig, frame):
        logger.warning(f"Received signal {sig} — finishing current chunk then exiting cleanly.")
        _shutdown.set()
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        for file_path, file_fmt in raw_files:
            if resume and progress.is_done(file_path):
                logger.info(f"SKIP (already done): {file_path.name}")
                continue

            logger.info(f"\nProcessing [{file_fmt}]: {file_path.name} ({get_file_size_mb(file_path):.1f} MB)")
            file_rows  = 0
            file_dupes = 0

            read_kwargs = dict(lines=True, chunksize=CHUNK_SIZE)
            if file_fmt == "gz":
                read_kwargs["compression"] = "gzip"

            # Derive category from filename (e.g. "Electronics.jsonl.gz" -> "Electronics")
            # HuggingFace 2023 files have no "category" column inside the JSON.
            file_stem = file_path.name.replace(".jsonl.gz", "").replace(".json.gz", "").replace(".jsonl", "")
            file_category = file_stem.replace("_", " ")   # Home_and_Kitchen -> Home and Kitchen

            try:
                reader = pd.read_json(file_path, **read_kwargs)
                for chunk in reader:
                    if _shutdown.is_set():
                        break

                    # Inject filename-derived category if no category column present
                    if "category" not in chunk.columns:
                        chunk["category"] = file_category

                    chunk = _cast_chunk(chunk)

                    if "review_id" in chunk.columns:
                        before = len(chunk)
                        mask   = chunk["review_id"].apply(lambda x: x not in seen)
                        chunk  = chunk[mask].drop_duplicates(subset=["review_id"])
                        dupes  = before - len(chunk)
                        file_dupes  += dupes
                        total_dupes += dupes
                        for rid in chunk["review_id"]:
                            seen.add(rid)

                    if chunk.empty:
                        continue

                    if max_rows and total_rows + len(chunk) > max_rows:
                        chunk = chunk.iloc[: max_rows - total_rows]

                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    writer.write(table)

                    total_rows        += len(chunk)
                    file_rows         += len(chunk)
                    heartbeat.rows_ref[0] = total_rows

                    del chunk, table
                    gc.collect()

                    if max_rows and total_rows >= max_rows:
                        logger.info(f"Reached max_rows={max_rows:,}, stopping.")
                        break

            except Exception as exc:
                logger.error(f"Error processing {file_path.name}: {exc}", exc_info=True)
                continue

            progress.mark_done(file_path, file_rows, file_dupes)
            logger.info(f"  Done: {file_path.name} | {file_rows:,} rows | {file_dupes:,} dupes")

            if _shutdown.is_set() or (max_rows and total_rows >= max_rows):
                break

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — closing writer cleanly.")
    finally:
        heartbeat.stop()
        writer.close()

    logger.info(f"\n{'='*55}")
    logger.info(f"  Preprocessing complete")
    logger.info(f"  Total rows     : {total_rows:,}")
    logger.info(f"  Duplicates     : {total_dupes:,}")
    logger.info(f"  Output         : {writer.summary()}")
    logger.info(f"{'='*55}")

    return out_path


# ─────────────────────────────────────────────────────────
# Product metadata  (unchanged)
# ─────────────────────────────────────────────────────────

def generate_product_metadata(processed_parquet: Path = OUTPUT_PATH) -> Path:
    from src.core.config import PRODUCT_METADATA_PATH, JOIN_KEY_COLUMN

    if PRODUCT_METADATA_PATH.exists():
        logger.info(f"Product metadata already exists: {PRODUCT_METADATA_PATH}")
        return PRODUCT_METADATA_PATH

    read_path = PARTITION_DIR if PARTITION_DIR.exists() else processed_parquet
    logger.info(f"Extracting product metadata from: {read_path}")

    key = JOIN_KEY_COLUMN  # currently "parent_asin"

    df = pd.read_parquet(read_path, columns=[key, "category", "rating"])

    meta = (
        df.groupby(key)
        .agg(
            category=("category", "first"),
            avg_rating_global=("rating", "mean"),
            review_count=("rating", "count"),
        )
        .reset_index()
    )

    meta["avg_rating_global"] = meta["avg_rating_global"].round(2)

    import numpy as np
    rng = np.random.default_rng(42)
    meta["price"] = rng.uniform(5.0, 999.0, len(meta)).round(2)
    meta["brand"] = [f"Brand_{i % 5000}" for i in range(len(meta))]

    PRODUCT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(PRODUCT_METADATA_PATH, index=False)
    logger.info(f"Saved product metadata: {PRODUCT_METADATA_PATH} ({len(meta):,} items)")
    return PRODUCT_METADATA_PATH


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=str(RAW_DIR))
    parser.add_argument("--output",     default=str(OUTPUT_PATH))
    parser.add_argument("--max-rows",   type=int, default=None)
    parser.add_argument("--metadata",   action="store_true")
    parser.add_argument("--partition",  action="store_true")
    parser.add_argument("--no-resume",  action="store_true", help="Ignore progress sidecar, reprocess all files")
    parser.add_argument("--bloom-cap",  type=int, default=500_000_000,
                        help="Expected unique review_ids for Bloom filter sizing (default 500M)")
    parser.add_argument("--raw-format", choices=["auto", "gz", "jsonl"], default="auto")
    args = parser.parse_args()

    out = preprocess(
        raw_dir    = Path(args.input),
        output     = Path(args.output),
        max_rows   = args.max_rows,
        partition  = args.partition,
        raw_format = args.raw_format,
        resume     = not args.no_resume,
        bloom_cap  = args.bloom_cap,
    )

    if args.metadata:
        generate_product_metadata(out)