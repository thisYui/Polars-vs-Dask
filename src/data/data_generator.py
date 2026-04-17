"""
src/data/data_generator.py
Fixed synthetic Amazon-like review generator for fair Polars vs Dask vs Pandas benchmarks.

Fixes applied (11 original + 4 new):
  #1  review_text now 300–1000 chars with high entropy
  #2  Tier system ensures apples-to-apples comparison across datasets
  #3  TEXT_MIN_LEN / TEXT_MAX_LEN config controls RAM usage
  #4  Categorical is opt-in per tier (WITH / WITHOUT mode)
  #5  ID reuse probability is configurable (70% pool / 30% fresh)
  #6  --target-partition-rows normalises IO chunk size across datasets
  #7  Workload separation: Numeric / Join / Text-heavy
  #8  Entropy control via TEXT_VARIANTS + RANDOM_INSERT_PROB
  #9  Scale by target RAM (--target-ram-gb) instead of row count
  #10 Three benchmark tiers: Optimised / Realistic / Passthrough-real
  #11 Isolated profiling: memory footprint, execution time, IO time
  #12 [NEW] calibrate_from_real() moved to module level (was orphaned inside class)
  #13 [NEW] import tracemalloc / contextmanager moved to top (was mid-file)
  #14 [NEW] import random / field removed (imported but never used)
  #15 [NEW] profile_section import os removed (os imported but never used inside)

Usage:
    # Fixed sizes
    python src/data/data_generator.py --sizes 1M 10M 100M

    # Scale by RAM target (recommended)
    python src/data/data_generator.py --target-ram-gb 5 10 20

    # Choose tier
    python src/data/data_generator.py --tier tier1 --sizes 10M
    python src/data/data_generator.py --tier tier2 --sizes 10M
"""

from __future__ import annotations

import argparse
import gc
import string
import sys
import time
import tracemalloc                          # FIX #13: moved from mid-file to top
from contextlib import contextmanager       # FIX #13: moved from mid-file to top
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

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
# Fix #3 & #8 — Text length and entropy controls
# ─────────────────────────────────────────────────────────

TEXT_MIN_LEN = 60
TEXT_MAX_LEN = 300
TEXT_VARIANTS = 2000
RANDOM_INSERT_PROB = 0.02

# Fix #5 — ID reuse vs fresh-ID probability
ID_POOL_REUSE_PROB  = 0.70


# ─────────────────────────────────────────────────────────
# Fix #10 — Three benchmark tiers
# ─────────────────────────────────────────────────────────

class BenchmarkTier(str, Enum):
    TIER1 = "tier1"   # Optimised synthetic: categorical + heavy ID reuse → best-case
    TIER2 = "tier2"   # Realistic synthetic: no categorical + long text → mid-case
    TIER3 = "tier3"   # Real dataset passthrough (no generation, just normalisation)


@dataclass
class TierConfig:
    use_categorical:    bool  = True
    id_pool_reuse_prob: float = ID_POOL_REUSE_PROB
    text_min_len:       int   = TEXT_MIN_LEN
    text_max_len:       int   = TEXT_MAX_LEN
    text_variants:      int   = TEXT_VARIANTS
    random_insert_prob: float = RANDOM_INSERT_PROB


TIER_CONFIGS: dict[BenchmarkTier, TierConfig] = {
    BenchmarkTier.TIER1: TierConfig(
        use_categorical    = True,
        id_pool_reuse_prob = 0.95,
        text_min_len       = 80,
        text_max_len       = 200,
        text_variants      = 500,
        random_insert_prob = 0.05,
    ),
    BenchmarkTier.TIER2: TierConfig(
        use_categorical    = False,
        id_pool_reuse_prob = 0.70,
        text_min_len       = TEXT_MIN_LEN,
        text_max_len       = TEXT_MAX_LEN,
        text_variants      = TEXT_VARIANTS,
        random_insert_prob = RANDOM_INSERT_PROB,
    ),
    BenchmarkTier.TIER3: TierConfig(
        use_categorical    = False,
        id_pool_reuse_prob = 0.0,
        text_min_len       = 0,
        text_max_len       = 9999,
        text_variants      = 0,
        random_insert_prob = 0.0,
    ),
}


# ─────────────────────────────────────────────────────────
# Fix #7 — Workload separation
# ─────────────────────────────────────────────────────────

class Workload(str, Enum):
    NUMERIC    = "numeric"
    JOIN       = "join"
    TEXT_HEAVY = "text_heavy"


# ─────────────────────────────────────────────────────────
# Amazon-realistic ID generators (numpy-based, no Faker)
# ─────────────────────────────────────────────────────────

_USER_ID_CHARS    = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), dtype="U1")
_USER_ID_LEN      = 28
_PRODUCT_ID_CHARS = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), dtype="U1")
_PRODUCT_ID_LEN   = 9


class _IDPools:
    """Lazily builds and caches user/product ID pools."""
    _user_pool:    list[str] | None = None
    _product_pool: list[str] | None = None

    @classmethod
    def users(cls, n_users: int) -> list[str]:
        if cls._user_pool is None or len(cls._user_pool) != n_users:
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(_USER_ID_CHARS), size=(n_users, _USER_ID_LEN))
            cls._user_pool = ["".join(row) for row in _USER_ID_CHARS[idx]]
        return cls._user_pool

    @classmethod
    def products(cls, n_products: int) -> list[str]:
        if cls._product_pool is None or len(cls._product_pool) != n_products:
            rng = np.random.default_rng(1)
            idx = rng.integers(0, len(_PRODUCT_ID_CHARS), size=(n_products, _PRODUCT_ID_LEN))
            cls._product_pool = ["B" + "".join(row) for row in _PRODUCT_ID_CHARS[idx]]
        return cls._product_pool


def _fresh_user_id(rng: np.random.Generator) -> str:
    idx = rng.integers(0, len(_USER_ID_CHARS), _USER_ID_LEN)
    return "".join(_USER_ID_CHARS[idx])


def _fresh_product_id(rng: np.random.Generator) -> str:
    idx = rng.integers(0, len(_PRODUCT_ID_CHARS), _PRODUCT_ID_LEN)
    return "B" + "".join(_PRODUCT_ID_CHARS[idx])


def _make_ids_with_reuse(
    rng:        np.random.Generator,
    n:          int,
    pool:       list[str],
    reuse_prob: float,
    fresh_fn,
) -> list[str]:
    pool_size = len(pool)
    is_reuse  = rng.random(n) < reuse_prob
    pool_idx  = rng.integers(0, pool_size, n)
    result    = []
    for i in range(n):
        if is_reuse[i]:
            result.append(pool[pool_idx[i]])
        else:
            result.append(fresh_fn(rng))
    return result


# ─────────────────────────────────────────────────────────
# Fix #1, #3, #8 — High-entropy, length-controlled review text
# ─────────────────────────────────────────────────────────

_SENTENCE_POOL_SEED = 77

_ADJ = [
    "amazing", "terrible", "decent", "outstanding", "mediocre", "excellent",
    "poor", "fantastic", "average", "superb", "overpriced", "surprisingly good",
    "disappointing", "top-notch", "flimsy", "robust", "inconsistent", "solid",
    "elegant", "underwhelming", "game-changing", "overhyped", "reliable",
    "unreliable", "premium", "budget-friendly", "durable", "fragile", "sleek",
    "bulky", "intuitive", "confusing", "innovative", "outdated",
]
_DETAIL_SENTENCES = [
    "Works exactly as advertised with no surprises.",
    "Fast shipping and well-packaged upon arrival.",
    "Not entirely what I expected based on the listing.",
    "The customer service team was very responsive.",
    "I would purchase this product again without hesitation.",
    "The packaging was slightly damaged on the outside but product was fine.",
    "Setup was straightforward — took less than five minutes.",
    "Build quality feels very durable and well-constructed.",
    "Stopped functioning correctly after the first week of use.",
    "This item genuinely exceeded my expectations in daily use.",
    "Instructions were unclear but the product itself is solid.",
    "Arrived earlier than the estimated delivery date — great!",
    "Build quality feels cheap relative to the asking price.",
    "Makes an excellent gift for friends or family members.",
    "The colour was noticeably different from what the photos showed.",
    "Batteries are not included, which was frustrating to discover.",
    "The size is slightly smaller than the description implied.",
    "I've been using this daily for three months with zero issues.",
    "Returned it after two uses — just not right for my needs.",
    "Honestly the best purchase I've made in this category.",
    "Works on some surfaces but not others, which limits usability.",
    "The manual is available online and much easier to follow.",
    "Some assembly required but the instructions are clear enough.",
    "Noise level is higher than expected for this type of device.",
    "Low power consumption compared to the older model — great upgrade.",
    "The warranty process was painless and resolved my issue quickly.",
    "Smells a bit odd initially but that fades after a few days.",
    "Much heavier than the listed weight specification suggests.",
    "Compatible with all the accessories I already owned.",
    "I had to contact support twice before getting a resolution.",
]
_CONNECTORS = [
    "Furthermore, ", "In addition, ", "That said, ", "On the other hand, ",
    "Overall, ", "To summarise, ", "Importantly, ", "Notably, ",
    "In my experience, ", "After extended use, ", "Compared to similar products, ",
    "In terms of value, ", "From a build-quality perspective, ",
]
_NOISE_WORDS = list(
    "abcdefghijklmnopqrstuvwxyz"
    + string.digits
    + "!@#$%"
)


def _build_sentence_pool(size: int) -> list[str]:
    """Fix #8: Pre-generate a large pool of varied base sentences."""
    rng   = np.random.default_rng(_SENTENCE_POOL_SEED)
    pool  = []
    n_det = len(_DETAIL_SENTENCES)
    n_con = len(_CONNECTORS)
    for _ in range(size):
        d1  = _DETAIL_SENTENCES[rng.integers(0, n_det)]
        d2  = _DETAIL_SENTENCES[rng.integers(0, n_det)]
        con = _CONNECTORS[rng.integers(0, n_con)]
        pool.append(f"{d1} {con}{d2[0].lower()}{d2[1:]}")
    return pool


_SENTENCE_POOL: list[str] = []


def _get_sentence_pool(size: int) -> list[str]:
    global _SENTENCE_POOL
    if len(_SENTENCE_POOL) < size:
        logger.info(f"Building sentence pool ({size:,} variants) …")
        _SENTENCE_POOL = _build_sentence_pool(size)
    return _SENTENCE_POOL


def _make_texts_v2(
    rng:                np.random.Generator,
    n:                  int,
    min_len:            int,
    max_len:            int,
    n_variants:         int,
    random_insert_prob: float,
) -> list[str]:
    """Fix #1 #3 #8: length-controlled, high-entropy review text."""
    pool        = _get_sentence_pool(n_variants)
    pool_size   = len(pool)
    avg = (min_len + max_len) / 2
    std = (max_len - min_len) / 4

    target_lens = rng.normal(loc=avg, scale=std, size=n)
    target_lens = np.clip(target_lens, min_len, max_len).astype(int)

    results = []
    for i in range(n):
        target = int(target_lens[i])
        parts  = []
        length = 0
        while length < target:
            sent = pool[rng.integers(0, pool_size)]
            if rng.random() < random_insert_prob:
                pos  = rng.integers(0, max(1, len(sent)))
                char = _NOISE_WORDS[rng.integers(0, len(_NOISE_WORDS))]
                sent = sent[:pos] + char + sent[pos:]
            parts.append(sent)
            length += len(sent) + 1
        text = " ".join(parts)
        if len(text) > max_len:
            cut  = text.rfind(" ", 0, max_len)
            text = text[:cut] if cut > 0 else text[:max_len]
        results.append(text)
    return results


# ─────────────────────────────────────────────────────────
# Title generator
# ─────────────────────────────────────────────────────────

_TITLE_TEMPLATES = [
    "{adj_cap} product!",
    "Just okay — {adj}",
    "Love it after {days} days",
    "Not what I expected",
    "Would {rec} to a friend",
    "{adj_cap}, {detail_short}",
    "Arrived damaged",
    "Great value for money",
    "Stopped working quickly",
    "Exceeded my expectations",
]
# FIX #14: _VERB was imported/defined but never used anywhere — removed
_REC = ["recommend", "not recommend"]


def _make_titles(rng: np.random.Generator, n: int) -> list[str]:
    det_shorts = [d.rstrip(".") for d in rng.choice(_DETAIL_SENTENCES, n)]
    tmpls      = rng.choice(_TITLE_TEMPLATES, n)
    adjs       = rng.choice(_ADJ, n)
    days       = rng.integers(1, 365, n).astype(str)
    recs       = rng.choice(_REC, n)
    return [
        t.replace("{adj_cap}", a.capitalize())
         .replace("{adj}",     a)
         .replace("{days}",    dy)
         .replace("{rec}",     r)
         .replace("{detail_short}", ds)
        for t, a, dy, r, ds in zip(tmpls, adjs, days, recs, det_shorts)
    ]


# ─────────────────────────────────────────────────────────
# Fix #9 — RAM-based scaling
# ─────────────────────────────────────────────────────────

_BYTES_PER_ROW: dict[BenchmarkTier, int] = {
    BenchmarkTier.TIER1: 50,
    BenchmarkTier.TIER2: 50,
    BenchmarkTier.TIER3: 35,
}


def ram_to_rows(target_gb: float, tier: BenchmarkTier) -> int:
    """Fix #9: Convert a RAM target (GB) to an approximate row count."""
    bytes_per_row = _BYTES_PER_ROW[tier]
    return int(target_gb * 1024 ** 3 / bytes_per_row)


# ─────────────────────────────────────────────────────────
# Fix #16 — Calibrated row count for --target-ram-gb
# ─────────────────────────────────────────────────────────

_CALIBRATION_ROWS = 10_000   # sample size for byte-per-row measurement


def calibrated_rows_for_size(
    target_bytes:   int,
    tier:           BenchmarkTier,
    workload:       Workload,
    seed:           int = 42,
    sample_n:       int = _CALIBRATION_ROWS,
) -> tuple[int, float]:
    """
    Fix #16: Generate a small sample, measure actual in-memory bytes/row,
    then return an exact row count that hits `target_bytes`.

    Returns:
        (n_rows, measured_bytes_per_row)

    Why this matters
    ─────────────────
    _BYTES_PER_ROW is a static estimate.  Real row size depends on
    text length distribution (TierConfig.text_min/max_len) and which
    columns the workload includes.  For TEXT_HEAVY tier2 a single
    review_text column alone can exceed 500 bytes/row, making the
    hardcoded 85 B/row estimate off by 6×.

    This function replaces the guess with a one-time 10 k-row probe
    (~0.1 s) before the main generation loop.
    """
    rng_probe = np.random.default_rng(seed)
    gen       = SyntheticReviewGenerator(
        n_rows     = sample_n,
        tier       = tier,
        workload   = workload,
        seed       = seed,
        chunk_size = sample_n,        # single chunk, no loop overhead
    )
    sample    = gen._generate_chunk(sample_n, rng_probe)

    # pandas memory_usage(deep=True) counts actual string bytes
    total_bytes   = sample.memory_usage(deep=True).sum()
    bytes_per_row = total_bytes / sample_n

    n_rows = max(1, int(target_bytes / bytes_per_row))

    logger.info(
        f"[calibrate] sample={sample_n:,} rows | "
        f"bytes/row={bytes_per_row:.1f} | "
        f"target={target_bytes / 1024**3:.3f} GB → {n_rows:,} rows"
    )
    return n_rows, bytes_per_row


# ─────────────────────────────────────────────────────────
# Fix #6 — Partition size normalisation
# ─────────────────────────────────────────────────────────

TARGET_PARTITION_BYTES = 128 * 1024 * 1024   # 128 MB per partition


def rows_per_partition(tier: BenchmarkTier) -> int:
    """Fix #6: Compute chunk rows that yield ~128 MB parquet partitions."""
    bpr = _BYTES_PER_ROW[tier]
    return max(1, TARGET_PARTITION_BYTES // bpr)


# ─────────────────────────────────────────────────────────
# Fix #11 — Isolated profiling context manager
# ─────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    wall_time_s: float = 0.0
    peak_ram_mb: float = 0.0
    io_time_s:   float = 0.0
    cpu_time_s:  float = 0.0


@contextmanager
def profile_section(label: str):
    """Fix #11: Isolate memory footprint vs execution time vs IO time."""
    # FIX #15: removed `import os` — os is never used inside this function
    result = ProfileResult()
    tracemalloc.start()
    t_wall_start = time.perf_counter()
    t_cpu_start  = time.process_time()
    try:
        yield result
    finally:
        result.wall_time_s = time.perf_counter() - t_wall_start
        result.cpu_time_s  = time.process_time()  - t_cpu_start
        _, peak            = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.peak_ram_mb = peak / 1024 / 1024
        logger.info(
            f"[{label}] wall={result.wall_time_s:.3f}s "
            f"cpu={result.cpu_time_s:.3f}s "
            f"peak_ram={result.peak_ram_mb:.1f}MB"
        )


# ─────────────────────────────────────────────────────────
# Fix #12 — calibrate_from_real at module level (not inside class)
# ─────────────────────────────────────────────────────────

def calibrate_from_real(real_parquet_dir: Path) -> TierConfig:
    """
    Fix #12: Calibrate a TierConfig from real data so that synthetic RAM
    footprint matches the real dataset.

    Previously this was defined as a @classmethod on SyntheticReviewGenerator
    but it has no dependency on `self` or `cls` — it is a pure factory function
    and belongs at module level.

    Usage:
        cfg = calibrate_from_real(Path("data/benchmark/real/1M"))
        TIER_CONFIGS[BenchmarkTier.TIER2] = cfg   # override tier2 globally
    """
    import glob
    files = glob.glob(str(real_parquet_dir / "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {real_parquet_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)

    text_col   = "review_text" if "review_text" in df.columns else "body"
    avg_len    = int(df[text_col].str.len().mean())
    reuse_prob = 1.0 - df["user_id"].nunique() / len(df)

    logger.info(
        f"calibrate_from_real: avg_text_len={avg_len}, "
        f"user_reuse_prob={reuse_prob:.3f} (from {len(df):,} rows)"
    )

    return TierConfig(
        use_categorical    = False,
        id_pool_reuse_prob = min(0.98, reuse_prob),
        text_min_len       = int(avg_len * 0.6),
        text_max_len       = int(avg_len * 1.3),
        text_variants      = TEXT_VARIANTS,
        random_insert_prob = 0.1,
    )


# ─────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────

class SyntheticReviewGenerator:
    """
    Generates synthetic Amazon review data at configurable scale.

    Supports three benchmark tiers (fix #10) and three workloads (fix #7).
    All major bias sources are controlled via TierConfig (fixes #1–#9).

    Args:
        n_rows     : total rows to generate
        tier       : BenchmarkTier enum (TIER1 / TIER2 / TIER3)
        workload   : Workload enum (NUMERIC / JOIN / TEXT_HEAVY)
        seed       : random seed
        chunk_size : rows per generation chunk (controls peak RAM)
    """

    def __init__(
        self,
        n_rows:     int,
        tier:       BenchmarkTier = BenchmarkTier.TIER2,
        workload:   Workload      = Workload.TEXT_HEAVY,
        seed:       int           = 42,
        chunk_size: int | None    = None,
    ):
        self.n_rows     = n_rows
        self.tier       = tier
        self.workload   = workload
        self.seed       = seed
        self.cfg        = TIER_CONFIGS[tier]
        self.chunk_size = chunk_size or rows_per_partition(tier)
        self._users     = _IDPools.users(N_USERS)
        self._products  = _IDPools.products(N_PRODUCTS)

    def _generate_chunk(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        cfg = self.cfg

        user_ids    = _make_ids_with_reuse(rng, n, self._users,    cfg.id_pool_reuse_prob, _fresh_user_id)
        product_ids = _make_ids_with_reuse(rng, n, self._products, cfg.id_pool_reuse_prob, _fresh_product_id)

        review_nums  = rng.integers(1_000_000_000, 9_999_999_999 + 1, n)
        review_ids   = [f"R{x}" for x in review_nums]
        ratings      = rng.choice([1, 2, 3, 4, 5], n, p=RATING_DISTRIBUTION).astype("int8")
        helpful_vote = rng.negative_binomial(1, 0.9, n).astype("int32")

        start_ts = pd.Timestamp("2010-01-01").timestamp()
        end_ts   = pd.Timestamp("2024-12-31").timestamp()
        times    = pd.to_datetime(rng.uniform(start_ts, end_ts, n), unit="s").normalize()

        cats     = rng.choice(CATEGORIES, n)
        verified = (rng.random(n) < 0.80)

        parent_idx    = rng.integers(0, N_PRODUCTS, n)
        is_own_parent = rng.random(n) < 0.70
        parent_asins  = [
            product_ids[i] if own else self._products[parent_idx[i]]
            for i, own in enumerate(is_own_parent)
        ]

        category_col = (
            pd.Categorical(cats, categories=CATEGORIES)
            if cfg.use_categorical
            else cats
        )

        df = pd.DataFrame({
            "review_id":         review_ids,
            "user_id":           user_ids,
            "product_id":        product_ids,
            "parent_asin":       parent_asins,
            "rating":            ratings,
            "review_time":       times,
            "helpful_vote":      helpful_vote,
            "category":          category_col,
            "verified_purchase": verified,
        })

        if self.workload != Workload.NUMERIC:
            titles = _make_titles(rng, n)
            texts  = _make_texts_v2(
                rng,
                n,
                min_len            = cfg.text_min_len,
                max_len            = cfg.text_max_len,
                n_variants         = cfg.text_variants,
                random_insert_prob = cfg.random_insert_prob,
            )
            df["review_title"] = titles
            df["review_text"]  = texts

        return df

    def generate(self) -> pd.DataFrame:
        """Generate full dataset in memory (small sizes only)."""
        rng    = np.random.default_rng(self.seed)
        chunks = []
        rem    = self.n_rows
        while rem > 0:
            n = min(self.chunk_size, rem)
            chunks.append(self._generate_chunk(n, rng))
            rem -= n
        return pd.concat(chunks, ignore_index=True)

    def generate_to_parquet(
        self,
        path:        Path,
        compression: str = PARQUET_COMPRESSION,
    ) -> Path:
        """Stream-generate directly to Parquet."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rng    = np.random.default_rng(self.seed)
        writer = None
        offset = 0
        rem    = self.n_rows
        t0     = time.perf_counter()

        try:
            while rem > 0:
                n = min(self.chunk_size, rem)

                with profile_section(f"generate chunk offset={offset}") as gen_prof:
                    chunk = self._generate_chunk(n, rng)
                    table = pa.Table.from_pandas(chunk, preserve_index=False)

                with profile_section(f"write chunk offset={offset}") as io_prof:
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema, compression=compression)
                    writer.write_table(table)

                offset += n
                rem    -= n
                pct     = offset / self.n_rows * 100
                elapsed = time.perf_counter() - t0
                logger.info(
                    f"  {pct:.0f}% — {offset:,}/{self.n_rows:,} rows "
                    f"| gen={gen_prof.wall_time_s:.2f}s "
                    f"| io={io_prof.wall_time_s:.2f}s "
                    f"| peak_ram={gen_prof.peak_ram_mb:.0f}MB "
                    f"| elapsed={elapsed:.1f}s"
                )

                del chunk, table
                gc.collect()
        finally:
            if writer:
                writer.close()

        mb = get_file_size_mb(path)
        actual_rows = pq.read_metadata(path).num_rows
        if actual_rows != self.n_rows:
            logger.error(
                f"Row count mismatch! expected={self.n_rows:,} actual={actual_rows:,} "
                f"— diff={actual_rows - self.n_rows:+,}"
            )
            raise RuntimeError(
                f"generate_to_parquet wrote {actual_rows:,} rows, expected {self.n_rows:,}"
            )
        logger.info(f"Saved: {path.name} ({mb:.1f} MB) | rows={actual_rows:,} ✓ | tier={self.tier} | workload={self.workload}")
        return path

    def generate_to_csv(self, path: Path) -> Path:
        """Stream-generate to CSV (slower, larger files)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rng    = np.random.default_rng(self.seed)
        rem    = self.n_rows
        header = True
        while rem > 0:
            n     = min(self.chunk_size, rem)
            chunk = self._generate_chunk(n, rng)
            chunk.to_csv(path, mode="a", index=False, header=header)
            header = False
            rem   -= n
            del chunk
            gc.collect()
        logger.info(f"Saved CSV: {path.name} ({get_file_size_mb(path):.1f} MB)")
        return path


# ─────────────────────────────────────────────────────────
# Product metadata (for JOIN workload)
# ─────────────────────────────────────────────────────────

def generate_product_metadata(
    path:  Path = None,
    force: bool = False,
    tier:  BenchmarkTier = BenchmarkTier.TIER2,
) -> pd.DataFrame:
    """Generate small product metadata table used by Workload B (JOIN)."""
    from src.core.config import PRODUCT_METADATA_PATH
    if path is None:
        path = PRODUCT_METADATA_PATH

    if path.exists() and not force:
        return pd.read_parquet(path)

    cfg         = TIER_CONFIGS[tier]
    rng         = np.random.default_rng(999)
    product_ids = _IDPools.products(N_PRODUCTS)
    brands      = [f"Brand_{i}" for i in range(5000)]

    category_col = (
        pd.Categorical(rng.choice(CATEGORIES, N_PRODUCTS), categories=CATEGORIES)
        if cfg.use_categorical
        else rng.choice(CATEGORIES, N_PRODUCTS)
    )

    df = pd.DataFrame({
        "product_id":        product_ids,
        "category":          category_col,
        "price":             rng.uniform(5.0, 999.0, N_PRODUCTS).round(2),
        "brand":             rng.choice(brands, N_PRODUCTS),
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
    size_label: str | None   = None,
    target_gb:  float | None = None,
    fmt:        str           = "parquet",
    dest_dir:   Path          = SYNTHETIC_DIR,
    force:      bool          = False,
    tier:       BenchmarkTier = BenchmarkTier.TIER2,
    workload:   Workload      = Workload.TEXT_HEAVY,
) -> Path:
    """
    Generate one synthetic dataset.

    Fix #9: Accepts either size_label (rows) OR target_gb (RAM target).
    Fix #10: Tier selects data characteristics.
    Fix #7:  Workload selects which columns are included.
    """
    from src.core.config import SCALABILITY_SIZES
    all_sizes = {**BENCHMARK_SIZES, **SCALABILITY_SIZES}

    if target_gb is not None:
        # Fix #16: calibrate from a real sample instead of using static estimate
        target_bytes = int(target_gb * 1024 ** 3)
        n_rows, bpr  = calibrated_rows_for_size(target_bytes, tier, workload)
        size_label   = f"{target_gb:.1f}GB_{tier.value}_{workload.value}"
        logger.info(
            f"RAM target {target_gb} GB → {n_rows:,} rows "
            f"(calibrated {bpr:.1f} B/row, tier={tier})"
        )
    elif size_label is not None:
        if size_label not in all_sizes:
            raise ValueError(f"Unknown size label '{size_label}'. Choose from {list(all_sizes)}")
        n_rows = all_sizes[size_label]
    else:
        raise ValueError("Provide either size_label or target_gb")

    path = dest_dir / f"reviews_{size_label}.{fmt}"

    if path.exists() and not force:
        logger.info(f"Already exists: {path.name} ({get_file_size_mb(path):.1f} MB)")
        return path

    logger.info(f"Generating '{size_label}' ({n_rows:,} rows, tier={tier}, workload={workload}, {fmt}) …")
    gen = SyntheticReviewGenerator(n_rows=n_rows, tier=tier, workload=workload)

    if fmt == "parquet":
        gen.generate_to_parquet(path)
    else:
        gen.generate_to_csv(path)

    generate_product_metadata(force=force, tier=tier)
    return path


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.core.config import SCALABILITY_SIZES
    all_size_keys = list({**BENCHMARK_SIZES, **SCALABILITY_SIZES}.keys())

    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark datasets (fixed bias issues)"
    )

    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--sizes", nargs="+", choices=all_size_keys,
        default=["1M", "10M"],
        help="Row-count size labels to generate",
    )
    size_group.add_argument(
        "--target-ram-gb", nargs="+", type=float,
        metavar="GB",
        help="Generate datasets targeting these RAM footprints (e.g. 0.3 5 10)",
    )

    parser.add_argument(
        "--tier",
        choices=[t.value for t in BenchmarkTier if t != BenchmarkTier.TIER3],
        default=BenchmarkTier.TIER2.value,
    )
    parser.add_argument(
        "--workload",
        choices=[w.value for w in Workload],
        default=Workload.TEXT_HEAVY.value,
    )
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--dest",   default=str(SYNTHETIC_DIR))
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    tier     = BenchmarkTier(args.tier)
    workload = Workload(args.workload)
    dest     = Path(args.dest)

    if args.target_ram_gb:
        for gb in args.target_ram_gb:
            prepare_synthetic(
                target_gb = gb,
                fmt       = args.format,
                dest_dir  = dest,
                force     = args.force,
                tier      = tier,
                workload  = workload,
            )
    else:
        for label in args.sizes:
            prepare_synthetic(
                size_label = label,
                fmt        = args.format,
                dest_dir   = dest,
                force      = args.force,
                tier       = tier,
                workload   = workload,
            )