"""
src/workloads/join.py
Workload 3: Join reviews with product metadata on product_id.
Tests large join performance.

Strategy: closure-based map_partitions broadcast join.

Why not reviews.merge(pandas_df) directly?
  dask-expr (Dask 2024.x) wraps any Pandas object in the expression tree
  with dd.from_pandas(), embedding it as a 'frompandas' task in the graph
  and serialising ~880 MB over TCP on every compute() call → OOM.

Why closure works:
  Dask uses cloudpickle to serialise the *function* (not its arguments).
  A closure capturing `meta` pickles to ~2.5 MB (the DataFrame itself),
  sent once per partition — not once per task × partition.
  With 1–2 partitions for 1M rows this is negligible.

Why not scheduler="synchronous"?
  Too conservative for this machine. The cluster can handle the join fine
  once we stop embedding DataFrames in the task graph.
"""

from pathlib import Path

from src.core.config import PRODUCT_METADATA_PATH, POLARS_STREAMING


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


def pandas_join(path: Path) -> "pd.DataFrame":
    import pandas as pd
    meta    = pd.read_parquet(_ensure_metadata())
    reviews = _read_parquet(path)
    result  = reviews.merge(meta, on="product_id", how="left")
    _       = len(result)
    return result


def polars_join(path: Path, lazy: bool = True) -> "pl.DataFrame":
    import polars as pl
    meta_path = _ensure_metadata()
    scan_path = f"{path}/*.parquet" if path.is_dir() else path
    if lazy:
        return (
            pl.scan_parquet(scan_path)
            .join(pl.scan_parquet(meta_path), on="product_id", how="left")
            .collect(streaming=POLARS_STREAMING)
        )
    return (
        pl.read_parquet(scan_path)
        .join(pl.read_parquet(meta_path), on="product_id", how="left")
    )


def dask_join(path: Path) -> "pd.DataFrame":
    """
    Practical Dask Join:
    - Manual Broadcast Join via map_partitions (prevents OOM on Windows)
    - No forced column list on reviews — let Dask infer schema from the actual
      parquet files to avoid 'column not in index' errors when SCHEMA_COLUMNS
      diverges from on-disk schema.
    - Join key is always "product_id", consistent with pandas_join / polars_join.
    """
    import dask
    import dask.dataframe as dd
    import pandas as pd
    import warnings

    JOIN_KEY = "product_id"

    # Silence scheduler warning definitively
    warnings.filterwarnings("ignore", message=".*single-machine scheduler.*")

    read_path = str(path / "*.parquet") if path.is_dir() else str(path)

    # Read without forcing a column list — schema inferred from parquet files.
    reviews = dd.read_parquet(read_path)

    # Broadcast the small metadata table (only columns needed post-join).
    meta = pd.read_parquet(_ensure_metadata(), columns=[JOIN_KEY, "price", "brand"])

    # Define output schema for Dask map_partitions.
    meta_out = reviews._meta.merge(meta.head(0), on=JOIN_KEY, how="left")

    def _local_merge(df, meta_df):
        return df.merge(meta_df, on=JOIN_KEY, how="left")

    # Use the threaded scheduler for stable RSS memory management on Windows.
    with dask.config.set({"dataframe.scheduler-warning": False}):
        return reviews.map_partitions(_local_merge, meta_df=meta, meta=meta_out).compute(scheduler="threads")