"""
src/workloads/filtering.py
Workload 1: Filter reviews where rating >= threshold.
Tests raw scan speed of each framework.
"""

from pathlib import Path

from src.core.config import FILTER_RATING_THRESHOLD, POLARS_STREAMING


def pandas_filter(path: Path) -> "pd.DataFrame":
    import pandas as pd
    df     = pd.read_parquet(path)
    result = df[df["rating"] >= FILTER_RATING_THRESHOLD]
    _      = len(result)   # force evaluation
    return result


def polars_filter(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    if lazy:
        return (
            pl.scan_parquet(path)
            .filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)
            .collect(streaming=POLARS_STREAMING)
        )
    df = pl.read_parquet(path)
    return df.filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)


def dask_filter(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd
    ddf    = dd.read_parquet(path)
    result = ddf[ddf["rating"] >= FILTER_RATING_THRESHOLD].compute()
    return result