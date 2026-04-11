"""
src/workloads/filtering.py
Workload 1: Filter reviews where rating >= threshold.
Tests raw scan speed of each framework.
"""

from pathlib import Path

from src.core.config import FILTER_RATING_THRESHOLD, POLARS_STREAMING


def _read_parquet(path: Path) -> "pd.DataFrame":
    """Đọc cả single file lẫn partition folder."""
    import pandas as pd
    if path.is_dir():
        files = sorted(path.glob("part-*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)

def pandas_filter(path: Path) -> "pd.DataFrame":
    import pandas as pd
    df     = _read_parquet(path)
    result = df[df["rating"] >= FILTER_RATING_THRESHOLD]
    _      = len(result)   # force evaluation
    return result


def pandas_filter(path: Path) -> "pd.DataFrame":
    import pandas as pd

    if path.is_dir():
        # Đọc từng file và concat — Pandas không có native folder read
        files = sorted(path.glob("part-*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = _read_parquet(path)

    return df[df["rating"] >= FILTER_RATING_THRESHOLD]


def polars_filter(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl

    # Polars tự handle cả file lẫn folder với glob pattern
    scan_path = f"{path}/*.parquet" if path.is_dir() else path

    if lazy:
        return (
            pl.scan_parquet(scan_path)
            .filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)
            .collect(streaming=POLARS_STREAMING)
        )
    return pl.read_parquet(scan_path).filter(pl.col("rating") >= FILTER_RATING_THRESHOLD)


def dask_filter(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd

    # Dask native support cả file lẫn folder
    read_path = str(path / "*.parquet") if path.is_dir() else str(path)
    ddf = dd.read_parquet(read_path)
    return ddf[ddf["rating"] >= FILTER_RATING_THRESHOLD].compute()