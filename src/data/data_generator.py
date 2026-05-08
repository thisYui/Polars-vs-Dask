"""
src/data/data_generator.py
Fixed synthetic Amazon-like review generator for fair Polars vs Dask vs Pandas benchmarks.

Fixes applied (11 original + 4 carried + 3 new directions):
  #1  review_text now 300–1000 chars with high entropy
  #2  Tier system ensures apples-to-apples comparison across datasets
  #3  TEXT_MIN_LEN / TEXT_MAX_LEN config controls RAM usage
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

Three generation directions (redesign):
  Hướng 1 — Real-like 100M (tier2):
      Giữ nguyên config TIER2 calibrated từ real data, scale lên 100M rows.
      Dùng: --tier tier2 --sizes 100M
      Mục tiêu: benchmark scalability với distribution trung thực.

  Hướng 2 — Skewed parent_asin 10M (tier2_skewed):
      Chỉ skew parent_asin: Zipf alpha = 3.0 (mạnh).
      Top 1% sản phẩm chiếm ~90% rows → mô phỏng viral/bestseller.
      user_zipf_alpha giữ nguyên TIER2 để isolate đúng biến.
      Dùng: --tier tier2_skewed --sizes 10M
      Mục tiêu: group-by/join trên cột cardinality thấp + skew nặng.

  Hướng 3 — High Unique user_id 10M (tier2_high_unique):
      id_pool_reuse_prob=0.50 → ~50% unique user_id.
      (trung bình mỗi user viết 2 review — light-user scenario)
      parent_asin giữ nguyên TIER2 để chỉ isolate biến user cardinality.
      user_zipf_alpha=0.8 → phân phối đều trong 50% unique pool.
      Dùng: --tier tier2_high_unique --sizes 10M
      Mục tiêu: dictionary encoding overhead, user-level group-by stress.

Usage:
    # Hướng 1 — Real-like, scale to 100M
    python src/data/data_generator.py --tier tier2 --sizes 100M

    # Hướng 2 — Skewed IDs, 10M
    python src/data/data_generator.py --tier tier2_skewed --sizes 10M

    # Hướng 3 — High unique IDs, 10M
    python src/data/data_generator.py --tier tier2_high_unique --sizes 10M

    # Classic usage
    python src/data/data_generator.py --sizes 1M 10M 100M
    python src/data/data_generator.py --target-ram-gb 5 10 20
    python src/data/data_generator.py --tier tier1 --sizes 10M
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.config import (
    N_PRODUCTS, N_PRODUCT_IDS, N_USERS,
    RATING_DISTRIBUTION,
    SYNTHETIC_DIR,
    BENCHMARK_SYN_DIR,
    PARQUET_COMPRESSION,
    BENCHMARK_SIZES, SCALABILITY_SIZES,
    TEXT_MIN_LEN as CONFIG_TEXT_MIN_LEN,
    TEXT_MAX_LEN as CONFIG_TEXT_MAX_LEN,
    TEXT_GENERATOR_MAX_LEN as CONFIG_TEXT_GENERATOR_MAX_LEN,
    TEXT_VARIANTS as CONFIG_TEXT_VARIANTS,
    RANDOM_INSERT_PROB as CONFIG_RANDOM_INSERT_PROB,
    ID_POOL_REUSE_PROB as CONFIG_ID_POOL_REUSE_PROB,
    PARENT_ASIN_REUSE_PROB,
    TEXT_LEN_LOGNORMAL_MU,
    TEXT_LEN_LOGNORMAL_SIGMA,
    TEXT_LEN_QUANTILE_POINTS,
    TEXT_LEN_QUANTILE_VALUES,
    USER_ZIPF_ALPHA,
    PARENT_ASIN_ZIPF_ALPHA,
    HELPFUL_ZERO_PROB,
    HELPFUL_LOGNORMAL_MEAN,
    HELPFUL_LOGNORMAL_SIGMA,
    HELPFUL_MAX,
)
from src.utils import get_logger, get_file_size_mb

logger = get_logger("data.generator")


# ─────────────────────────────────────────────────────────
# Calibrated generation controls
# ─────────────────────────────────────────────────────────
# Values come from src/core/config.py, calibrated from real/1M.

TEXT_MIN_LEN = CONFIG_TEXT_MIN_LEN       # p5  = 12  chars (calibrated from real/1M)
TEXT_MAX_LEN = CONFIG_TEXT_MAX_LEN       # p85 = 374 chars (calibrated from real/1M)
TEXT_GENERATOR_MAX_LEN = CONFIG_TEXT_GENERATOR_MAX_LEN
TEXT_VARIANTS = CONFIG_TEXT_VARIANTS
RANDOM_INSERT_PROB = CONFIG_RANDOM_INSERT_PROB

# ID reuse vs fresh-ID probability
ID_POOL_REUSE_PROB = CONFIG_ID_POOL_REUSE_PROB


# ─────────────────────────────────────────────────────────
# Fix #10 — Three benchmark tiers
# ─────────────────────────────────────────────────────────

class BenchmarkTier(str, Enum):
    TIER1            = "tier1"           # Optimised synthetic: categorical + heavy ID reuse → best-case
    TIER2            = "tier2"           # Realistic synthetic: calibrated from real data → mid-case (10M / 100M)
    TIER2_SKEWED     = "tier2_skewed"    # [NEW] Skewed IDs: Zipf alpha=2.0, hot-key pressure (10M)
    TIER2_HIGH_UNIQUE= "tier2_high_unique"  # [NEW] High unique IDs: low reuse (~80% unique) (10M)
    TIER3            = "tier3"           # Real dataset passthrough (no generation, just normalisation)


@dataclass
class TierConfig:
    id_pool_reuse_prob: float = ID_POOL_REUSE_PROB
    parent_asin_reuse_prob: float = PARENT_ASIN_REUSE_PROB
    text_min_len:       int   = TEXT_MIN_LEN
    text_max_len:       int   = TEXT_MAX_LEN
    text_variants:      int   = TEXT_VARIANTS
    random_insert_prob: float = RANDOM_INSERT_PROB
    user_zipf_alpha:        float = USER_ZIPF_ALPHA
    parent_asin_zipf_alpha: float = PARENT_ASIN_ZIPF_ALPHA

    # Real-data calibrated text length model.
    text_len_lognormal_mu:    float = TEXT_LEN_LOGNORMAL_MU
    text_len_lognormal_sigma: float = TEXT_LEN_LOGNORMAL_SIGMA
    use_text_quantile_model:   bool = True

    # Real-data calibrated helpful_vote model.
    helpful_zero_prob:        float = HELPFUL_ZERO_PROB
    helpful_lognormal_mean:   float = HELPFUL_LOGNORMAL_MEAN
    helpful_lognormal_sigma:  float = HELPFUL_LOGNORMAL_SIGMA
    helpful_max:              int   = HELPFUL_MAX


TIER_CONFIGS: dict[BenchmarkTier, TierConfig] = {
    BenchmarkTier.TIER1: TierConfig(
        id_pool_reuse_prob = 0.95,
        text_min_len       = 80,
        text_max_len       = 200,
        text_variants      = 500,
        random_insert_prob = 0.05,
        use_text_quantile_model = False,
    ),
    BenchmarkTier.TIER2: TierConfig(
        id_pool_reuse_prob = ID_POOL_REUSE_PROB,
        parent_asin_reuse_prob = PARENT_ASIN_REUSE_PROB,
        text_min_len       = TEXT_MIN_LEN,
        text_max_len       = TEXT_GENERATOR_MAX_LEN,
        text_variants      = TEXT_VARIANTS,
        random_insert_prob = RANDOM_INSERT_PROB,
        user_zipf_alpha         = USER_ZIPF_ALPHA,
        parent_asin_zipf_alpha  = PARENT_ASIN_ZIPF_ALPHA,
        text_len_lognormal_mu    = TEXT_LEN_LOGNORMAL_MU,
        text_len_lognormal_sigma = TEXT_LEN_LOGNORMAL_SIGMA,
        use_text_quantile_model  = True,
        helpful_zero_prob        = HELPFUL_ZERO_PROB,
        helpful_lognormal_mean   = HELPFUL_LOGNORMAL_MEAN,
        helpful_lognormal_sigma  = HELPFUL_LOGNORMAL_SIGMA,
        helpful_max              = HELPFUL_MAX,
    ),
    # ── [NEW] Hướng 2: Skewed parent_asin ───────────────────────────────────
    # Chỉ skew parent_asin với Zipf alpha=3.0 (mạnh):
    #   top 1% sản phẩm chiếm ~90% rows → mô phỏng viral/bestseller product.
    # user_zipf_alpha giữ nguyên TIER2 (~1.1) để isolate đúng biến.
    # Mục tiêu: kiểm tra group-by/join trên cột có cardinality thấp + skew nặng.
    BenchmarkTier.TIER2_SKEWED: TierConfig(
        id_pool_reuse_prob       = ID_POOL_REUSE_PROB,
        parent_asin_reuse_prob   = PARENT_ASIN_REUSE_PROB,
        text_min_len             = TEXT_MIN_LEN,
        text_max_len             = TEXT_GENERATOR_MAX_LEN,
        text_variants            = TEXT_VARIANTS,
        random_insert_prob       = RANDOM_INSERT_PROB,
        user_zipf_alpha          = USER_ZIPF_ALPHA,        # giữ nguyên TIER2
        parent_asin_zipf_alpha   = 3.0,   # <<< chỉ skew parent_asin: top 1% ~ 90% rows
        text_len_lognormal_mu    = TEXT_LEN_LOGNORMAL_MU,
        text_len_lognormal_sigma = TEXT_LEN_LOGNORMAL_SIGMA,
        use_text_quantile_model  = True,
        helpful_zero_prob        = HELPFUL_ZERO_PROB,
        helpful_lognormal_mean   = HELPFUL_LOGNORMAL_MEAN,
        helpful_lognormal_sigma  = HELPFUL_LOGNORMAL_SIGMA,
        helpful_max              = HELPFUL_MAX,
    ),

    # ── [NEW] Hướng 3: High Unique IDs ───────────────────────────────────────
    # id_pool_reuse_prob=0.50 → ~50% unique user_id
    #   (trung bình mỗi user viết 2 review — gần giống real Amazon light-users).
    # parent_asin_reuse_prob giữ nguyên TIER2 để chỉ isolate biến user cardinality.
    # user_zipf_alpha=0.8 → phân phối đều hơn giữa các unique user (không có superstar).
    # Mục tiêu: dictionary encoding overhead, high-cardinality user group-by.
    BenchmarkTier.TIER2_HIGH_UNIQUE: TierConfig(
        id_pool_reuse_prob       = 0.50,  # <<< 50% unique users (TIER2 ~0.70 reuse)
        parent_asin_reuse_prob   = PARENT_ASIN_REUSE_PROB,  # giữ nguyên TIER2
        text_min_len             = TEXT_MIN_LEN,
        text_max_len             = TEXT_GENERATOR_MAX_LEN,
        text_variants            = TEXT_VARIANTS,
        random_insert_prob       = RANDOM_INSERT_PROB,
        user_zipf_alpha          = 0.8,   # <<< đều hơn trong 50% unique pool
        parent_asin_zipf_alpha   = PARENT_ASIN_ZIPF_ALPHA,  # giữ nguyên TIER2
        text_len_lognormal_mu    = TEXT_LEN_LOGNORMAL_MU,
        text_len_lognormal_sigma = TEXT_LEN_LOGNORMAL_SIGMA,
        use_text_quantile_model  = True,
        helpful_zero_prob        = HELPFUL_ZERO_PROB,
        helpful_lognormal_mean   = HELPFUL_LOGNORMAL_MEAN,
        helpful_lognormal_sigma  = HELPFUL_LOGNORMAL_SIGMA,
        helpful_max              = HELPFUL_MAX,
    ),

    BenchmarkTier.TIER3: TierConfig(
        id_pool_reuse_prob = 0.0,
        text_min_len       = 0,
        text_max_len       = 9999,
        text_variants      = 0,
        random_insert_prob = 0.0,
        use_text_quantile_model = False,
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


_ZIPF_WEIGHT_CACHE: dict[tuple[int, float], np.ndarray] = {}


def _zipf_rank_weights(pool_size: int, alpha: float) -> np.ndarray:
    """Rank-frequency weights where probability(rank) is proportional to rank^-alpha."""
    key = (pool_size, round(alpha, 6))
    weights = _ZIPF_WEIGHT_CACHE.get(key)
    if weights is None:
        ranks = np.arange(1, pool_size + 1, dtype="float64")
        weights = ranks ** (-alpha)
        weights /= weights.sum()
        _ZIPF_WEIGHT_CACHE[key] = weights
    return weights


def _make_ids_with_target_cardinality(
    rng: np.random.Generator,
    n: int,
    pool: list[str],
    alpha: float,
    emitted_unique: int,
    target_unique: int,
) -> tuple[list[str], int]:
    """
    Draw IDs from a calibrated Zipf-like pool while guaranteeing target cardinality.

    The old reuse/fresh split matched a rough reuse probability but doubled user
    cardinality at 1M rows. This keeps the frozen unique counts from 01a/01c and
    still produces hot-key skew through rank-frequency sampling.
    """
    target_unique = max(1, min(target_unique, len(pool)))
    guaranteed_n = min(n, max(0, target_unique - emitted_unique))

    ids: list[str] = []
    if guaranteed_n:
        ids.extend(pool[emitted_unique: emitted_unique + guaranteed_n])
        emitted_unique += guaranteed_n

    sampled_n = n - guaranteed_n
    if sampled_n:
        weights = _zipf_rank_weights(target_unique, alpha)
        idx = rng.choice(target_unique, size=sampled_n, replace=True, p=weights)
        ids.extend(pool[i] for i in idx)

    if guaranteed_n and sampled_n:
        rng.shuffle(ids)
    return ids, emitted_unique


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
    lognormal_mu:       float,
    lognormal_sigma:    float,
    use_quantile_model: bool = True,
) -> list[str]:
    """
    Length-controlled, high-entropy review text.

    Fixes:
    - Use calibrated log-normal target lengths instead of normal+clip.
    - Truncate each row to its own target length instead of global max_len.
    """
    pool      = _get_sentence_pool(n_variants)
    pool_size = len(pool)

    if use_quantile_model:
        quantiles = np.array([0.0, *TEXT_LEN_QUANTILE_POINTS, 1.0], dtype="float64")
        lengths = np.array([min_len, *TEXT_LEN_QUANTILE_VALUES, max_len], dtype="float64")
        target_lens = np.interp(rng.random(n), quantiles, lengths)
    else:
        target_lens = rng.lognormal(
            mean=lognormal_mu,
            sigma=lognormal_sigma,
            size=n,
        )
    target_lens = np.clip(target_lens, min_len, max_len).astype(int)

    results: list[str] = []
    for i in range(n):
        target = max(1, int(target_lens[i]))
        parts: list[str] = []
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

        if len(text) > target:
            text = text[:target].rstrip()

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
    BenchmarkTier.TIER1:             50,
    BenchmarkTier.TIER2:             50,
    BenchmarkTier.TIER2_SKEWED:      50,   # same text profile as TIER2
    BenchmarkTier.TIER2_HIGH_UNIQUE: 50,   # same text profile as TIER2
    BenchmarkTier.TIER3:             35,
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
    Fix #16 (improved): Generate a small sample, write to a temp Parquet file,
    read it back, then measure actual in-memory RAM bytes/row.

    Returns:
        (n_rows, measured_bytes_per_row)

    Why the original was wrong
    ──────────────────────────
    The previous version measured `memory_usage(deep=True)` on the freshly
    generated DataFrame.  That measures Python object RAM *during generation*,
    but when a Parquet file is read back by Pandas/PyArrow the string columns
    land in Arrow StringArray buffers whose per-row overhead is different
    (often 2–3× higher for short-to-medium strings due to Arrow offset arrays
    and validity bitmaps on top of the raw character data).

    Result: the old code underestimated bytes/row by ~2.4× for tier2/TEXT_HEAVY,
    so --target-ram-gb 5 produced only ~2.1 GB instead of 5 GB.

    Fix: write the sample to a temporary Parquet file, read it back with
    pd.read_parquet(), then measure memory_usage(deep=True) on *that* DataFrame.
    This captures the real post-read RAM footprint that benchmarks will see.
    """
    import tempfile

    rng_probe = np.random.default_rng(seed)
    gen       = SyntheticReviewGenerator(
        n_rows     = sample_n,
        tier       = tier,
        workload   = workload,
        seed       = seed,
        chunk_size = sample_n,        # single chunk, no loop overhead
    )
    sample = gen._generate_chunk(sample_n, rng_probe)

    # FIX #17: Write sample to temp Parquet then read back to measure
    # the *actual* RAM footprint that pd.read_parquet() produces,
    # not the in-memory generation footprint (which is ~2–3× smaller).
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        table = pa.Table.from_pandas(sample, preserve_index=False)
        pq.write_table(table, tmp_path, compression=PARQUET_COMPRESSION)
        del sample, table

        sample_readback = pd.read_parquet(tmp_path)
        total_bytes     = sample_readback.memory_usage(deep=True).sum()
        bytes_per_row   = total_bytes / sample_n
        del sample_readback
    finally:
        tmp_path.unlink(missing_ok=True)

    n_rows = max(1, int(target_bytes / bytes_per_row))

    logger.info(
        f"[calibrate] sample={sample_n:,} rows | "
        f"bytes/row={bytes_per_row:.1f} (post-parquet read) | "
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

    Calibration strategy (aligned with 01b_calibrate_synthetic.ipynb):
    - text_min_len  = p5  of real text_len  (loại outlier ngắn)
    - text_max_len  = p99 of real text_len  (preserve validation tail)
    - lognormal params fit trực tiếp từ log(text_len)
    - id_pool_reuse_prob calibrated riêng cho user_id (không dùng avg của user+item)

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

    text_col = "review_text" if "review_text" in df.columns else "body"
    text_len = df[text_col].fillna("").astype(str).str.len()

    # Text length calibration — p5/p99 keeps the real long-tail visible enough
    # for the 01c KL/Wasserstein validation while still clipping extreme outliers.
    text_min_len = int(text_len.quantile(0.05))
    text_max_len = int(text_len.quantile(0.99))

    # Log-normal fit on log(text_len) — reference for validation
    log_len          = np.log(text_len.clip(lower=1))
    lognormal_mu     = float(log_len.mean())
    lognormal_sigma  = float(log_len.std())

    # ID reuse — calibrate user_id separately (item reuse is much lower)
    total_rows        = len(df)
    reuse_prob_user   = 1.0 - df["user_id"].nunique() / total_rows
    reuse_prob_parent = 1.0 - df["parent_asin"].nunique() / total_rows

    logger.info(
        f"calibrate_from_real: rows={total_rows:,} | "
        f"text p5={text_min_len} p99={text_max_len} | "
        f"lognormal μ={lognormal_mu:.4f} σ={lognormal_sigma:.4f} | "
        f"user_reuse_prob={reuse_prob_user:.4f} | "
        f"parent_reuse_prob={reuse_prob_parent:.4f}"
    )

    return TierConfig(
        id_pool_reuse_prob       = min(0.98, reuse_prob_user),
        parent_asin_reuse_prob   = min(0.98, reuse_prob_parent),
        text_min_len             = text_min_len,
        text_max_len             = text_max_len,
        text_variants            = TEXT_VARIANTS,
        random_insert_prob       = RANDOM_INSERT_PROB,
        text_len_lognormal_mu    = lognormal_mu,
        text_len_lognormal_sigma = lognormal_sigma,
    )


# ─────────────────────────────────────────────────────────
# Helpful vote generator
# ─────────────────────────────────────────────────────────

def _make_helpful_votes(
    rng: np.random.Generator,
    n: int,
    zero_prob: float,
    lognormal_mean: float,
    lognormal_sigma: float,
    max_value: int,
) -> np.ndarray:
    """
    Generate helpful_vote as a zero-inflated long-tail variable.

    Real data behavior:
    - Most reviews have 0 helpful votes.
    - Upper quantiles still have a visible tail, e.g. p90≈2, p95≈4, p99≈16.
    """
    values = np.zeros(n, dtype="int32")
    nonzero_mask = rng.random(n) >= zero_prob
    tail_n = int(nonzero_mask.sum())

    if tail_n > 0:
        tail = rng.lognormal(
            mean=lognormal_mean,
            sigma=lognormal_sigma,
            size=tail_n,
        )
        tail = np.rint(tail).astype("int32")
        tail = np.clip(tail, 1, max_value).astype("int32")
        values[nonzero_mask] = tail

    return values


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
        self._product_ids = _IDPools.products(N_PRODUCT_IDS)
        self._parent_asins = _IDPools.products(N_PRODUCTS)
        if tier == BenchmarkTier.TIER2:
            # Calibrated: unique count = min(pool, n_rows) → matches real data cardinality
            self._target_unique_users = min(N_USERS, n_rows)
            self._target_unique_parent_asins = min(N_PRODUCTS, n_rows)
        elif tier == BenchmarkTier.TIER2_HIGH_UNIQUE:
            # Hướng 3: ~50% unique users (reuse_prob=0.50 → unique_prob=0.50)
            # parent_asin giữ nguyên TIER2 (full cardinality)
            self._target_unique_users = min(N_USERS, max(1, round(n_rows * (1.0 - self.cfg.id_pool_reuse_prob))))
            self._target_unique_parent_asins = min(N_PRODUCTS, n_rows)  # TIER2-style
        elif tier == BenchmarkTier.TIER2_SKEWED:
            # Hướng 2: user cardinality TIER2-style (full), parent_asin TIER2-style (full)
            # Skew được tạo bởi Zipf alpha=3.0 trên parent_asin, không phải bằng pool nhỏ
            self._target_unique_users = min(N_USERS, n_rows)
            self._target_unique_parent_asins = min(N_PRODUCTS, n_rows)
        else:
            self._target_unique_users = min(N_USERS, max(1, round(n_rows * (1.0 - self.cfg.id_pool_reuse_prob))))
            self._target_unique_parent_asins = min(N_PRODUCTS, max(1, round(n_rows * (1.0 - self.cfg.parent_asin_reuse_prob))))
        self._target_unique_products = min(N_PRODUCT_IDS, n_rows)
        self._emitted_unique_users = 0
        self._emitted_unique_products = 0
        self._emitted_unique_parent_asins = 0

    def _generate_chunk(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        cfg = self.cfg

        user_ids, self._emitted_unique_users = _make_ids_with_target_cardinality(
            rng,
            n,
            self._users,
            cfg.user_zipf_alpha,
            self._emitted_unique_users,
            self._target_unique_users,
        )
        product_ids, self._emitted_unique_products = _make_ids_with_target_cardinality(
            rng,
            n,
            self._product_ids,
            cfg.parent_asin_zipf_alpha,
            self._emitted_unique_products,
            self._target_unique_products,
        )
        parent_asins, self._emitted_unique_parent_asins = _make_ids_with_target_cardinality(
            rng,
            n,
            self._parent_asins,
            cfg.parent_asin_zipf_alpha,
            self._emitted_unique_parent_asins,
            self._target_unique_parent_asins,
        )

        review_nums  = rng.integers(1_000_000_000, 9_999_999_999 + 1, n)
        review_ids   = [f"R{x}" for x in review_nums]
        ratings      = rng.choice([1, 2, 3, 4, 5], n, p=RATING_DISTRIBUTION).astype("int8")
        helpful_vote = _make_helpful_votes(
            rng,
            n,
            zero_prob       = cfg.helpful_zero_prob,
            lognormal_mean  = cfg.helpful_lognormal_mean,
            lognormal_sigma = cfg.helpful_lognormal_sigma,
            max_value       = cfg.helpful_max,
        )

        start_ts = pd.Timestamp("2010-01-01").timestamp()
        end_ts   = pd.Timestamp("2024-12-31").timestamp()
        times    = pd.to_datetime(rng.uniform(start_ts, end_ts, n), unit="s").normalize()

        verified = (rng.random(n) < 0.80)

        df = pd.DataFrame({
            "review_id":         review_ids,
            "user_id":           user_ids,
            "product_id":        product_ids,
            "parent_asin":       parent_asins,
            "rating":            ratings,
            "review_time":       times,
            "helpful_vote":      helpful_vote,
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
                lognormal_mu       = cfg.text_len_lognormal_mu,
                lognormal_sigma    = cfg.text_len_lognormal_sigma,
                use_quantile_model = cfg.use_text_quantile_model,
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
) -> pd.DataFrame:
    """Generate small product metadata table used by Workload B (JOIN)."""
    from src.core.config import PRODUCT_METADATA_PATH
    if path is None:
        path = PRODUCT_METADATA_PATH

    if path.exists() and not force:
        return pd.read_parquet(path)

    rng         = np.random.default_rng(999)
    product_ids = _IDPools.products(N_PRODUCTS)
    brands      = [f"Brand_{i}" for i in range(5000)]

    df = pd.DataFrame({
        "product_id":        product_ids,
        "parent_asin":       product_ids,
        "price":             rng.uniform(5.0, 999.0, N_PRODUCTS).round(2),
        "brand":             rng.choice(brands, N_PRODUCTS),
        "avg_rating_global": rng.uniform(1.0, 5.0, N_PRODUCTS).round(2),
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Product metadata: {path.name} ({len(df):,} rows)")
    return df


# ─────────────────────────────────────────────────────────
# GB-label helpers
# ─────────────────────────────────────────────────────────

def _gb_label(target_gb: float) -> str:
    """Convert float GB to canonical label: 5.0 → '5GB', 5.5 → '5.5GB'."""
    return f"{int(target_gb)}GB" if target_gb == int(target_gb) else f"{target_gb}GB"


def _split_to_benchmark_syn(
    source_path:    Path,
    gb_label:       str,
    partition:      bool = False,
    target_file_mb: int  = 128,
    force:          bool = False,
) -> Path:
    """
    Copy / partition a generated synthetic file into
    data/benchmark_syn/<XGB>/ so that benchmark runners can find it
    using the same path convention as row-count splits.

    When partition=False  → benchmark_syn/5GB/data.parquet   (single file)
    When partition=True   → benchmark_syn/5GB/part-000.parquet ... (multi-file)
    """
    import gc
    import math
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest_dir = BENCHMARK_SYN_DIR / gb_label
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not partition:
        dest_file = dest_dir / "data.parquet"
        if dest_file.exists() and not force:
            logger.info(f"  [{gb_label}] already exists (single-file) — skip")
            return dest_dir
        import shutil
        shutil.copy2(source_path, dest_file)
        size_mb = dest_file.stat().st_size / 1024 ** 2
        logger.info(f"  [{gb_label}] → {dest_file} ({size_mb:.0f} MB)")
        return dest_dir

    # ── Partitioned mode ──────────────────────────────────
    existing = list(dest_dir.glob("part-*.parquet"))
    if existing and not force:
        total_mb = sum(p.stat().st_size for p in existing) / 1024 ** 2
        logger.info(
            f"  [{gb_label}] partitions already exist "
            f"({len(existing)} files, {total_mb:.0f} MB) — skip"
        )
        return dest_dir

    # Estimate rows_per_partition from source file size
    source_size  = source_path.stat().st_size
    source_rows  = pq.read_metadata(source_path).num_rows
    bytes_per_row = (source_size / max(source_rows, 1)) * 1.2   # 20% safety factor
    rows_per_part = max(
        100_000,
        int((target_file_mb * 1024 ** 2) / bytes_per_row),
    )
    n_parts = math.ceil(source_rows / rows_per_part)
    logger.info(
        f"  [{gb_label}] partitioning {source_rows:,} rows → "
        f"~{n_parts} files × ~{rows_per_part:,} rows (~{target_file_mb} MB each)"
    )

    pf        = pq.ParquetFile(source_path)
    schema    = pq.read_schema(source_path)
    writer    = None
    part_idx  = 0
    part_rows = 0
    total_written = 0

    try:
        for batch in pf.iter_batches(batch_size=200_000):
            table  = pa.Table.from_batches([batch])
            offset = 0
            while offset < len(table):
                if writer is None:
                    part_path = dest_dir / f"part-{part_idx:03d}.parquet"
                    writer    = pq.ParquetWriter(part_path, schema, compression="snappy")
                    part_rows = 0

                space = rows_per_part - part_rows
                take  = min(space, len(table) - offset)
                writer.write_table(table.slice(offset, take))
                part_rows     += take
                total_written += take
                offset        += take

                if part_rows >= rows_per_part:
                    writer.close()
                    written_path = dest_dir / f"part-{part_idx:03d}.parquet"
                    size_mb = written_path.stat().st_size / 1024 ** 2
                    logger.info(f"    ✓ part-{part_idx:03d}.parquet | {part_rows:,} rows | {size_mb:.0f} MB")
                    writer   = None
                    part_idx += 1

            del table
            gc.collect()
    finally:
        if writer:
            writer.close()
            written_path = dest_dir / f"part-{part_idx:03d}.parquet"
            size_mb = written_path.stat().st_size / 1024 ** 2
            logger.info(f"    ✓ part-{part_idx:03d}.parquet | {part_rows:,} rows | {size_mb:.0f} MB")

    total_mb = sum(p.stat().st_size for p in dest_dir.glob("part-*.parquet")) / 1024 ** 2
    n_files  = len(list(dest_dir.glob("part-*.parquet")))
    logger.info(f"  [{gb_label}] done → {dest_dir} | {total_written:,} rows | {n_files} files | {total_mb:.0f} MB")
    return dest_dir


# ─────────────────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────────────────

def prepare_synthetic(
    size_label:     str | None   = None,
    target_gb:      float | None = None,
    fmt:            str           = "parquet",
    dest_dir:       Path          = SYNTHETIC_DIR,
    force:          bool          = False,
    tier:           BenchmarkTier = BenchmarkTier.TIER2,
    workload:       Workload      = Workload.TEXT_HEAVY,
    # ── NEW: split into benchmark_syn/<XGB>/ after generating ──
    split_benchmark: bool = False,
    partition:       bool = False,
    target_file_mb:  int  = 128,
) -> Path:
    """
    Generate one synthetic dataset.

    Fix #9: Accepts either size_label (rows) OR target_gb (RAM target).
    Fix #10: Tier selects data characteristics.
    Fix #7:  Workload selects which columns are included.

    When target_gb is given and split_benchmark=True, the generated file is
    also copied / partitioned into data/benchmark_syn/<XGB>/ so benchmark
    runners can find it immediately without a separate split step.
    """
    from src.core.config import SCALABILITY_SIZES
    all_sizes = {**BENCHMARK_SIZES, **SCALABILITY_SIZES}

    # ── [Hướng 1] Fallback: nếu config.py chưa có "100M", thêm hardcoded ──
    # SCALABILITY_SIZES trong config.py cũ có thể chưa có 100M.
    # Thêm ở đây để generate_syn_100m step không crash.
    _DIRECTION_SIZES: dict[str, int] = {
        "10M":  10_000_000,   # Hướng 2 & 3
        "100M": 100_000_000,  # Hướng 1
    }
    for k, v in _DIRECTION_SIZES.items():
        all_sizes.setdefault(k, v)

    if target_gb is not None:
        target_bytes = int(target_gb * 1024 ** 3)
        n_rows, bpr  = calibrated_rows_for_size(target_bytes, tier, workload)
        size_label   = f"{target_gb:.1f}GB_{tier.value}_{workload.value}"
        logger.info(
            f"RAM target {target_gb} GB → {n_rows:,} rows "
            f"(calibrated {bpr:.1f} B/row, tier={tier})"
        )
    elif size_label is not None:
        # Allow custom labels (e.g. "10M_skewed") that are not in all_sizes dict.
        # Strip a known suffix to resolve the base row count.
        _base_label = size_label
        for _suffix in ("_skewed", "_highuid", f"_{tier.value}"):
            if _base_label.endswith(_suffix):
                _base_label = _base_label[: -len(_suffix)]
                break
        if _base_label in all_sizes:
            n_rows = all_sizes[_base_label]
        elif size_label in all_sizes:
            n_rows = all_sizes[size_label]
        else:
            raise ValueError(f"Unknown size label '{size_label}'. Choose from {list(all_sizes)}")
    else:
        raise ValueError("Provide either size_label or target_gb")

    path = dest_dir / f"reviews_{size_label}.{fmt}"

    if path.exists() and not force:
        logger.info(f"Already exists: {path.name} ({get_file_size_mb(path):.1f} MB)")
    else:
        logger.info(f"Generating '{size_label}' ({n_rows:,} rows, tier={tier}, workload={workload}, {fmt}) …")
        gen = SyntheticReviewGenerator(n_rows=n_rows, tier=tier, workload=workload)

        if fmt == "parquet":
            gen.generate_to_parquet(path)
        else:
            gen.generate_to_csv(path)

        generate_product_metadata(force=force)

    # ── NEW: auto-split into benchmark_syn/<label>/ when requested ──
    # Supports both --target-ram-gb (GB label) and --sizes (e.g. "10M_skewed", "100M")
    if split_benchmark:
        if target_gb is not None:
            bench_label = _gb_label(target_gb)
        else:
            bench_label = size_label          # e.g. "10M_skewed", "100M"
        logger.info(f"Splitting into benchmark_syn/{bench_label}/ …")
        _split_to_benchmark_syn(
            source_path    = path,
            gb_label       = bench_label,
            partition      = partition,
            target_file_mb = target_file_mb,
            force          = force,
        )

    return path


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.core.config import SCALABILITY_SIZES
    # Đảm bảo 10M và 100M luôn available trong CLI kể cả khi config cũ chưa có
    _all_size_keys_dict = {**BENCHMARK_SIZES, **SCALABILITY_SIZES}
    _all_size_keys_dict.setdefault("10M",  10_000_000)
    _all_size_keys_dict.setdefault("100M", 100_000_000)
    all_size_keys = list(_all_size_keys_dict.keys())

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
        help=(
            "tier1=optimised (high reuse), "
            "tier2=realistic/real-like (default, dùng cho 100M), "
            "tier2_skewed=hot-key skew Zipf α=2.0 (10M), "
            "tier2_high_unique=~80%% unique IDs (10M)"
        ),
    )
    parser.add_argument(
        "--workload",
        choices=[w.value for w in Workload],
        default=Workload.TEXT_HEAVY.value,
    )
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--dest",   default=str(SYNTHETIC_DIR))
    parser.add_argument("--force",  action="store_true")
    # ── NEW: benchmark_syn split options ──
    parser.add_argument(
        "--split-benchmark", action="store_true",
        help=(
            "After generating, copy/partition file into "
            "data/benchmark_syn/<label>/ (works with both --sizes and --target-ram-gb)"
        ),
    )
    parser.add_argument(
        "--size-label",
        default=None,
        metavar="LABEL",
        help=(
            "Custom output label for the generated file, e.g. '10M_skewed'. "
            "Row count is inferred from the base label (strip suffix). "
            "Used by run_pipeline for Hướng 2 & 3 so files don't overwrite each other."
        ),
    )
    parser.add_argument(
        "--partition", action="store_true",
        help="When --split-benchmark: write multi-file partitions instead of single file",
    )
    parser.add_argument(
        "--target-file-mb", type=int, default=128,
        help="Target MB per partition file when --partition is used (default: 128)",
    )
    args = parser.parse_args()

    tier     = BenchmarkTier(args.tier)
    workload = Workload(args.workload)
    dest     = Path(args.dest)

    if args.target_ram_gb:
        for gb in args.target_ram_gb:
            prepare_synthetic(
                target_gb        = gb,
                fmt              = args.format,
                dest_dir         = dest,
                force            = args.force,
                tier             = tier,
                workload         = workload,
                split_benchmark  = args.split_benchmark,
                partition        = args.partition,
                target_file_mb   = args.target_file_mb,
            )
    else:
        for label in args.sizes:
            # --size-label overrides the default label (useful for a single --sizes entry)
            effective_label = args.size_label if (args.size_label and len(args.sizes) == 1) else label
            prepare_synthetic(
                size_label      = effective_label,
                fmt             = args.format,
                dest_dir        = dest,
                force           = args.force,
                tier            = tier,
                workload        = workload,
                split_benchmark = args.split_benchmark,
                partition       = args.partition,
                target_file_mb  = args.target_file_mb,
            )