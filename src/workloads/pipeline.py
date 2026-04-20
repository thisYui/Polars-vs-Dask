"""
src/workloads/pipeline.py
Workload 4: Complex multi-step pipeline.
  filter → groupby → join → sort
Simulates a realistic analytics query.

Fix:
  - dask_pipeline: was using bare `path` for dd.read_parquet — fails silently
    when --partition is used (folder instead of single file).
  - polars_pipeline: scan_path now uses glob pattern for partition folders,
    consistent with filtering.py and groupby.py.
"""

from pathlib import Path

from src.core.config import (
    FILTER_RATING_THRESHOLD, GROUPBY_COLUMN,
    PRODUCT_METADATA_PATH, POLARS_STREAMING,
)


def _read_parquet(path: Path) -> "pd.DataFrame":
    """Đọc cả single file lẫn partition folder."""
    import pandas as pd
    if path.is_dir():
        files = sorted(path.glob("part-*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def _ensure_metadata() -> Path:
    if not PRODUCT_METADATA_PATH.exists():
        from src.data.data_generator import generate_product_metadata
        generate_product_metadata()
    return PRODUCT_METADATA_PATH


def pandas_pipeline(path: Path) -> "pd.DataFrame":
    import pandas as pd
    meta = pd.read_parquet(_ensure_metadata(), columns=["product_id", "price", "brand"])
    df   = _read_parquet(path)
    return (
        df[df["rating"] >= FILTER_RATING_THRESHOLD]
        .groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .merge(meta, on="product_id", how="left")
        .sort_values("count", ascending=False)
    )


def polars_pipeline(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    meta_path = _ensure_metadata()

    # Handle partition folder
    scan_path = f"{path}/*.parquet" if path.is_dir() else path

    aggs = [
        pl.col("rating").mean().alias("rating_mean"),
        pl.col("rating").count().alias("rating_count"),
        pl.col("rating").sum().alias("rating_sum"),
    ]
    if lazy:
        meta = pl.scan_parquet(meta_path).select(["product_id", "price", "brand"])
        return (
            pl.scan_parquet(scan_path)
            .filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)
            .group_by(GROUPBY_COLUMN)
            .agg(aggs)
            .join(meta, on="product_id", how="left")
            .sort("rating_count", descending=True)
            .collect(streaming=POLARS_STREAMING)
        )
    meta = pl.read_parquet(meta_path).select(["product_id", "price", "brand"])
    return (
        pl.read_parquet(scan_path)
        .filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)
        .group_by(GROUPBY_COLUMN)
        .agg(aggs)
        .join(meta, on="product_id", how="left")
        .sort("rating_count", descending=True)
    )


def dask_pipeline(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd
    import pandas as pd
    from src.core.config import GROUPBY_COLUMN

    meta = pd.read_parquet(_ensure_metadata(), columns=[GROUPBY_COLUMN, "price", "brand"])

    # Fix: handle partition folder
    read_path = str(path / "*.parquet") if path.is_dir() else str(path)
    
    # ❗ Pushdown: Filter and GroupBy on a subset of columns
    reviews = dd.read_parquet(read_path, columns=[GROUPBY_COLUMN, "rating"])

    # Build the lazy pipeline: Filter -> GroupBy -> Aggregation
    pipeline = (
        reviews[reviews["rating"] >= FILTER_RATING_THRESHOLD]
        .groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    
    # ❗ Stay in Dask! Use map_partitions for the final join instead of intermediate compute.
    # We sort at the end after compute because sorting is expensive in distributed Dask 
    # and usually done as the last step in a report.
    def _local_join(df, meta_df):
        return df.merge(meta_df, on=GROUPBY_COLUMN, how="left")

    # Distributed compute + final sort
    return (
        pipeline.map_partitions(_local_join, meta_df=meta)
        .compute()
        .sort_values("count", ascending=False)
    )