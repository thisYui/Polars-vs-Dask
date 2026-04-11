"""
workloads.py
Four benchmark workloads implemented for all three frameworks.

Each workload has three variants:
    run_pandas_<workload>(path) -> result
    run_polars_<workload>(path) -> result
    run_dask_<workload>(path)   -> result

All functions accept a dataset path (Parquet recommended).
They return the computed result so the caller can force execution
and verify correctness.
"""

from pathlib import Path

from config import (
    FILTER_RATING_THRESHOLD,
    GROUPBY_COLUMN,
    PRODUCT_METADATA_PATH,
    POLARS_STREAMING,
    DASK_PARTITION_SIZE,
)
from utils import get_logger

logger = get_logger(__name__)


# ═════════════════════════════════════════════
#  WORKLOAD 1: FILTER
#  rating >= FILTER_RATING_THRESHOLD
# ═════════════════════════════════════════════

def run_pandas_filter(path: Path) -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_parquet(path)
    result = df[df["rating"] >= FILTER_RATING_THRESHOLD]
    # Force evaluation (already eager)
    _ = len(result)
    return result


def run_polars_filter(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    if lazy:
        result = (
            pl.scan_parquet(path)
            .filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)
            .collect(streaming=POLARS_STREAMING)
        )
    else:
        df = pl.read_parquet(path)
        result = df.filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)
    return result


def run_dask_filter(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd
    ddf = dd.read_parquet(path)
    result = ddf[ddf["rating"] >= FILTER_RATING_THRESHOLD].compute()
    return result


# ═════════════════════════════════════════════
#  WORKLOAD 2: GROUPBY AGGREGATION
#  group by product_id → mean/count/sum rating
# ═════════════════════════════════════════════

def run_pandas_groupby(path: Path) -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_parquet(path)
    result = (
        df.groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    return result


def run_polars_groupby(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    if lazy:
        result = (
            pl.scan_parquet(path)
            .group_by(GROUPBY_COLUMN)
            .agg([
                pl.col("rating").mean().alias("rating_mean"),
                pl.col("rating").count().alias("rating_count"),
                pl.col("rating").sum().alias("rating_sum"),
            ])
            .collect(streaming=POLARS_STREAMING)
        )
    else:
        df = pl.read_parquet(path)
        result = df.group_by(GROUPBY_COLUMN).agg([
            pl.col("rating").mean().alias("rating_mean"),
            pl.col("rating").count().alias("rating_count"),
            pl.col("rating").sum().alias("rating_sum"),
        ])
    return result


def run_dask_groupby(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd
    ddf = dd.read_parquet(path)
    result = (
        ddf.groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .compute()
    )
    return result


# ═════════════════════════════════════════════
#  WORKLOAD 3: JOIN
#  reviews JOIN product_metadata ON product_id
# ═════════════════════════════════════════════

def _check_metadata() -> Path:
    if not PRODUCT_METADATA_PATH.exists():
        from data_generator import generate_product_metadata
        generate_product_metadata(PRODUCT_METADATA_PATH)
    return PRODUCT_METADATA_PATH


def run_pandas_join(path: Path) -> "pd.DataFrame":
    import pandas as pd
    meta_path = _check_metadata()
    reviews = pd.read_parquet(path)
    metadata = pd.read_parquet(meta_path)
    result = reviews.merge(metadata, on="product_id", how="left")
    _ = len(result)
    return result


def run_polars_join(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    meta_path = _check_metadata()
    if lazy:
        reviews = pl.scan_parquet(path)
        metadata = pl.scan_parquet(meta_path)
        result = (
            reviews.join(metadata, on="product_id", how="left")
            .collect(streaming=POLARS_STREAMING)
        )
    else:
        reviews = pl.read_parquet(path)
        metadata = pl.read_parquet(meta_path)
        result = reviews.join(metadata, on="product_id", how="left")
    return result


def run_dask_join(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd
    import pandas as pd
    meta_path = _check_metadata()
    reviews = dd.read_parquet(path)
    # Metadata is small → broadcast join (bring into memory)
    metadata = pd.read_parquet(meta_path)
    result = reviews.merge(metadata, on="product_id", how="left").compute()
    return result


# ═════════════════════════════════════════════
#  WORKLOAD 4: COMPLEX PIPELINE
#  filter → groupby → join → sort
# ═════════════════════════════════════════════

def run_pandas_pipeline(path: Path) -> "pd.DataFrame":
    import pandas as pd
    meta_path = _check_metadata()

    df = pd.read_parquet(path)
    metadata = pd.read_parquet(meta_path)

    result = (
        df[df["rating"] >= FILTER_RATING_THRESHOLD]                 # filter
        .groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()                                               # groupby
        .merge(metadata[["product_id", "price", "brand"]],
               on="product_id", how="left")                         # join
        .sort_values("count", ascending=False)                      # sort
    )
    return result


def run_polars_pipeline(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    meta_path = _check_metadata()

    if lazy:
        reviews = pl.scan_parquet(path)
        metadata = pl.scan_parquet(meta_path).select(
            ["product_id", "price", "brand"]
        )
        result = (
            reviews
            .filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)   # filter
            .group_by(GROUPBY_COLUMN)
            .agg([
                pl.col("rating").mean().alias("rating_mean"),
                pl.col("rating").count().alias("rating_count"),
                pl.col("rating").sum().alias("rating_sum"),
            ])                                                       # groupby
            .join(metadata, on="product_id", how="left")            # join
            .sort("rating_count", descending=True)                  # sort
            .collect(streaming=POLARS_STREAMING)
        )
    else:
        df = pl.read_parquet(path)
        metadata = pl.read_parquet(meta_path).select(
            ["product_id", "price", "brand"]
        )
        result = (
            df.filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)
            .group_by(GROUPBY_COLUMN)
            .agg([
                pl.col("rating").mean().alias("rating_mean"),
                pl.col("rating").count().alias("rating_count"),
                pl.col("rating").sum().alias("rating_sum"),
            ])
            .join(metadata, on="product_id", how="left")
            .sort("rating_count", descending=True)
        )
    return result


def run_dask_pipeline(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd
    import pandas as pd
    meta_path = _check_metadata()

    reviews = dd.read_parquet(path)
    metadata = pd.read_parquet(meta_path)[["product_id", "price", "brand"]]

    result = (
        reviews[reviews["rating"] >= FILTER_RATING_THRESHOLD]       # filter
        .groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()                                               # groupby
        .compute()                                                   # materialise before pandas join
    )

    result = (
        result
        .merge(metadata, on="product_id", how="left")               # join
        .sort_values("count", ascending=False)                      # sort
    )
    return result


# ═════════════════════════════════════════════
#  Dispatch Table
# ═════════════════════════════════════════════

WORKLOAD_FNS = {
    "pandas": {
        "filter":   run_pandas_filter,
        "groupby":  run_pandas_groupby,
        "join":     run_pandas_join,
        "pipeline": run_pandas_pipeline,
    },
    "polars": {
        "filter":   run_polars_filter,
        "groupby":  run_polars_groupby,
        "join":     run_polars_join,
        "pipeline": run_polars_pipeline,
    },
    "dask": {
        "filter":   run_dask_filter,
        "groupby":  run_dask_groupby,
        "join":     run_dask_join,
        "pipeline": run_dask_pipeline,
    },
}


def get_workload_fn(framework: str, workload: str):
    """Return the workload function for a framework/workload combination."""
    try:
        return WORKLOAD_FNS[framework][workload]
    except KeyError:
        raise ValueError(
            f"Unknown framework '{framework}' or workload '{workload}'. "
            f"Available frameworks: {list(WORKLOAD_FNS)}; "
            f"workloads: {list(WORKLOAD_FNS['pandas'])}"
        )
