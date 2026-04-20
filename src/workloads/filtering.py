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
    from src.core.config import SCHEMA_COLUMNS

    # Silence scheduler warning definitively
    warnings.filterwarnings("ignore", message=".*single-machine scheduler.*")

    read_path = str(path / "*.parquet") if path.is_dir() else str(path)

    # Pushdown: Always good practice to reduce I/O.
    # We let Dask handle partitioning naturally based on the Parquet file structure.
    ddf = dd.read_parquet(read_path, columns=SCHEMA_COLUMNS)

    # Using 'threads' scheduler for RSS memory stability on Windows 16GB.
    return ddf[ddf["rating"] >= FILTER_RATING_THRESHOLD].compute(scheduler="threads")