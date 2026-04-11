"""
src/workloads/__init__.py
Central registry mapping (framework, workload) → callable.

Usage:
    from src.workloads import WORKLOAD_REGISTRY, get_workload_fn

    fn = get_workload_fn("pandas", "groupby")
    fn = get_workload_fn("polars_lazy", "filter")
    fn = get_workload_fn("dask", "pipeline")
"""

from src.workloads.filtering import pandas_filter, polars_filter, dask_filter
from src.workloads.groupby   import pandas_groupby, polars_groupby, dask_groupby
from src.workloads.join      import pandas_join, polars_join, dask_join
from src.workloads.pipeline  import pandas_pipeline, polars_pipeline, dask_pipeline

import functools


def _polars_eager(fn):
    """Wrap a polars workload function to force eager mode."""
    @functools.wraps(fn)
    def wrapper(path):
        return fn(path, lazy=False)
    return wrapper


def _polars_lazy(fn):
    """Wrap a polars workload function to force lazy mode (default)."""
    @functools.wraps(fn)
    def wrapper(path):
        return fn(path, lazy=True)
    return wrapper


# ─────────────────────────────────────────────────────────
# Registry: framework_label → workload_name → callable
# ─────────────────────────────────────────────────────────

WORKLOAD_REGISTRY: dict[str, dict[str, callable]] = {
    "pandas": {
        "filter":   pandas_filter,
        "groupby":  pandas_groupby,
        "join":     pandas_join,
        "pipeline": pandas_pipeline,
    },
    "polars_lazy": {
        "filter":   _polars_lazy(polars_filter),
        "groupby":  _polars_lazy(polars_groupby),
        "join":     _polars_lazy(polars_join),
        "pipeline": _polars_lazy(polars_pipeline),
    },
    "polars_eager": {
        "filter":   _polars_eager(polars_filter),
        "groupby":  _polars_eager(polars_groupby),
        "join":     _polars_eager(polars_join),
        "pipeline": _polars_eager(polars_pipeline),
    },
    # Convenience alias: "polars" → lazy by default
    "polars": {
        "filter":   _polars_lazy(polars_filter),
        "groupby":  _polars_lazy(polars_groupby),
        "join":     _polars_lazy(polars_join),
        "pipeline": _polars_lazy(polars_pipeline),
    },
    "dask": {
        "filter":   dask_filter,
        "groupby":  dask_groupby,
        "join":     dask_join,
        "pipeline": dask_pipeline,
    },
}

# ── All valid keys ────────────────────────────────────────
FRAMEWORKS = list(WORKLOAD_REGISTRY.keys())
WORKLOADS  = ["filter", "groupby", "join", "pipeline"]


def get_workload_fn(framework: str, workload: str) -> callable:
    """
    Retrieve a workload function by framework and workload name.

    Args:
        framework : one of FRAMEWORKS (e.g. "pandas", "polars_lazy", "dask")
        workload  : one of WORKLOADS  (e.g. "filter", "groupby")

    Raises:
        KeyError if framework or workload is not registered.
    """
    if framework not in WORKLOAD_REGISTRY:
        raise KeyError(
            f"Unknown framework '{framework}'. "
            f"Available: {FRAMEWORKS}"
        )
    fw_registry = WORKLOAD_REGISTRY[framework]
    if workload not in fw_registry:
        raise KeyError(
            f"Workload '{workload}' not found for framework '{framework}'. "
            f"Available: {list(fw_registry)}"
        )
    return fw_registry[workload]


__all__ = [
    "WORKLOAD_REGISTRY",
    "FRAMEWORKS",
    "WORKLOADS",
    "get_workload_fn",
]