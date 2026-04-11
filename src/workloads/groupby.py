"""
src/workloads/groupby.py
Workload 2: GroupBy aggregation — group by product_id, compute mean/count/sum of rating.
Tests aggregation engine performance.
"""

from pathlib import Path

from src.core.config import GROUPBY_COLUMN, POLARS_STREAMING


def pandas_groupby(path: Path) -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_parquet(path)
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