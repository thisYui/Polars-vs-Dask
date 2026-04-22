"""
src/workloads/filtering.py
Workload 1: Filter reviews where rating >= threshold.
Tests raw scan speed of each framework.

Fix:
  - Removed duplicate pandas_filter definition (Python was silently using
    the second one, dropping the len() force-evaluation from the first).
  - Merged both into one clean definition with force evaluation.
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
    df     = _read_parquet(path)
    result = df[df["rating"] >= FILTER_RATING_THRESHOLD]
    _      = len(result)   # force evaluation
    return result


def polars_filter(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl

    # Polars handles both single file and folder natively
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
    import warnings

    # Silence scheduler warning definitively
    warnings.filterwarnings("ignore", message=".*single-machine scheduler.*")

    read_path = str(path / "*.parquet") if path.is_dir() else str(path)

    # Read without forcing a column list — let Dask infer schema from the
    # actual parquet files to avoid 'column not in index' errors when
    # SCHEMA_COLUMNS diverges from the on-disk schema.
    ddf = dd.read_parquet(read_path)

    # Using 'threads' scheduler for RSS memory stability on Windows 16GB.
    return ddf[ddf["rating"] >= FILTER_RATING_THRESHOLD].compute(scheduler="threads")