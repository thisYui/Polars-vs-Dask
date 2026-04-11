"""
src/workloads/groupby.py
Workload 2: GroupBy aggregation — group by product_id, compute mean/count/sum of rating.
Tests aggregation engine performance.
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
    import pandas as pd
    df = _read_parquet(path)
    return (
        df.groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )


def polars_groupby(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    aggs = [
        pl.col("rating").mean().alias("rating_mean"),
        pl.col("rating").count().alias("rating_count"),
        pl.col("rating").sum().alias("rating_sum"),
    ]
    if lazy:
        return (
            pl.scan_parquet(path)
            .group_by(GROUPBY_COLUMN)
            .agg(aggs)
            .collect(streaming=POLARS_STREAMING)
        )
    return pl.read_parquet(path).group_by(GROUPBY_COLUMN).agg(aggs)


def dask_groupby(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd
    ddf = dd.read_parquet(path)
    return (
        ddf.groupby(GROUPBY_COLUMN)["rating"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .compute()
    )