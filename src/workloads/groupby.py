"""
src/workloads/groupby.py
Workload 2: GroupBy aggregation — group by product_id, compute mean/count/sum of rating.
Tests aggregation engine performance.

Fix:
  - dask_groupby: was using bare `path` instead of glob pattern for
    partition folders, causing Dask to fail silently or read only one file
    when --partition is used.
"""

from pathlib import Path

from src.core.config import GROUPBY_COLUMN, POLARS_STREAMING


def _read_parquet(path: Path) -> "pd.DataFrame":
    """Đọc cả single file lẫn partition folder."""
    import pandas as pd
    if path.is_dir():
        files = sorted(path.glob("part-*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def pandas_groupby(path: Path) -> "pd.DataFrame":
    df = _read_parquet(path)
    return (
        df.groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )


def polars_groupby(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl

    scan_path = f"{path}/*.parquet" if path.is_dir() else path

    aggs = [
        pl.col("rating").mean().alias("rating_mean"),
        pl.col("rating").count().alias("rating_count"),
        pl.col("rating").sum().alias("rating_sum"),
    ]
    if lazy:
        return (
            pl.scan_parquet(scan_path)
            .group_by(GROUPBY_COLUMN)
            .agg(aggs)
            .collect(streaming=POLARS_STREAMING)
        )
    return pl.read_parquet(scan_path).group_by(GROUPBY_COLUMN).agg(aggs)


def dask_groupby(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd

    # Fix: handle partition folder the same way as filter/join
    read_path = str(path / "*.parquet") if path.is_dir() else str(path)
    ddf = dd.read_parquet(read_path)
    return (
        ddf.groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .compute()
    )