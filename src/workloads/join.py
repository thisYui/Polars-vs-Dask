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
    Elite Dask Join: 
    - Column Pushdown
    - Index-based Join (Hash reuse)
    - Manual Broadcast via map_partitions with explicit metadata
    - Repartitioning for core utilization
    """
    import dask
    import dask.dataframe as dd
    import pandas as pd
    import psutil
    import warnings
    from src.core.config import GROUPBY_COLUMN, SCHEMA_COLUMNS

    # ❗ Silence scheduler warning definitively
    warnings.filterwarnings("ignore", message=".*single-machine scheduler.*")

    read_path = str(path / "*.parquet") if path.is_dir() else str(path)
    n_cores = psutil.cpu_count(logical=False) or 4

    # ❗ Pushdown + Repartition
    reviews = dd.read_parquet(read_path, columns=SCHEMA_COLUMNS).repartition(npartitions=n_cores * 2)
    
    # ❗ Micro-opt: set_index on metadata table to reuse hash for each partition
    meta = pd.read_parquet(_ensure_metadata(), columns=[GROUPBY_COLUMN, "price", "brand"]).set_index(GROUPBY_COLUMN)

    # ❗ Explicit Metadata: Tell Dask exactly what the output schema looks like
    meta_out = reviews._meta.merge(meta.head(0), left_on=GROUPBY_COLUMN, right_index=True, how="left")

    def _local_join(df, meta_df):
        return df.join(meta_df, on=GROUPBY_COLUMN, how="left")

    # ❗ Silence scheduler warning and use threads for stability
    with dask.config.set({"dataframe.scheduler-warning": False}):
        return reviews.map_partitions(_local_join, meta_df=meta, meta=meta_out).compute(scheduler="threads")