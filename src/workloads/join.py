"""
src/workloads/join.py
Workload 3: Join reviews with product metadata on product_id.
Tests large join performance.
"""

from pathlib import Path

from src.core.config import PRODUCT_METADATA_PATH, POLARS_STREAMING


def _ensure_metadata() -> Path:
    if not PRODUCT_METADATA_PATH.exists():
        from src.data.data_generator import generate_product_metadata
        generate_product_metadata()
    return PRODUCT_METADATA_PATH


def pandas_join(path: Path) -> "pd.DataFrame":
    import pandas as pd
    meta    = pd.read_parquet(_ensure_metadata())
    reviews = pd.read_parquet(path)
    result  = reviews.merge(meta, on="product_id", how="left")
    _       = len(result)
    return result


def polars_join(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    meta_path = _ensure_metadata()
    if lazy:
        return (
            pl.scan_parquet(path)
            .join(pl.scan_parquet(meta_path), on="product_id", how="left")
            .collect(streaming=POLARS_STREAMING)
        )
    return (
        pl.read_parquet(path)
        .join(pl.read_parquet(meta_path), on="product_id", how="left")
    )


def dask_join(path: Path) -> "pd.DataFrame":
    import dask.dataframe as dd
    import pandas as pd
    # Metadata is small → broadcast (no shuffle needed)
    meta    = pd.read_parquet(_ensure_metadata())
    reviews = dd.read_parquet(path)
    return reviews.merge(meta, on="product_id", how="left").compute()