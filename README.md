```
project_root/
│
├── benchmarks/
│   ├── dask_run.py
│   ├── pandas_run.py
│   ├── polars_run.py
│   └── run_all.py
│
├── configs/
│   ├── logical.yaml
│   ├── physical.yaml
│   └── validation.yaml
│
├── data/
│
├── logs/
│
├── notebooks/
│       │
│       ├── 00_setup_and_verify.ipynb
│       ├── data_prep/
│       │   ├── 01a_verify_real_data.ipynb
│       │   ├── 01b_calibrate_synthetic.ipynb
│       │   └── 01c_validate_synthetic.ipynb
│       ├── benchmarks/
│       │   ├── 02_logical_scaling.ipynb
│       │   ├── 03_physical_scaling.ipynb
│       │   └── 04_real_vs_synthetic_runtime.ipynb
│       ├── analysis/
│       │   ├── 05_workload_breakdown.ipynb
│       │   └── 06_lazy_vs_eager_polars.ipynb
│       └── report/
│           └── 09_final_report.ipynb
│
├── results/
│   ├── plots/
│   │
│   ├── raw/
│   │    ├── dask_100m_results.csv
│   │    ├── dask_colab_results.csv
│   │    ├── dask_highuid_skewed_results.csv
│   │    ├── dask_real_results.csv
│   │    ├── dask_size_results.csv
│   │    ├── pandas_colab_results.csv
│   │    ├── pandas_highuid_skewed_results.csv
│   │    ├── pandas_real_results.csv
│   │    ├── pandas_size_results.csv
│   │    ├── polars_100m_results.csv
│   │    ├── polars_colab_results.csv
│   │    ├── polars_eager_colab_results.csv
│   │    ├── polars_eager_highuid_skewed_results.csv
│   │    ├── polars_eager_real_results.csv
│   │    ├── polars_eager_size_results.csv
│   │    ├── polars_highuid_skewed_results.csv
│   │    ├── polars_real_results.csv
│   │    └── polars_size_results.csv
│   │
│   └── tables/
│
├── scripts/
│   ├── run_pipeline.py
│   └── test.ipynb
│
└── src/
    ├── core/
    │   ├── __init__.py
    │   ├── benchmark.py
    │   ├── experiment_config.py
    │   ├── config.py
    │   ├── memory_profiler.py
    │   └── timer.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── compress_jsonl.py
    │   ├── data_generator.py
    │   ├── download_amazon.py
    │   ├── preprocess_amazon.py
    │   └── split_dataset.py
    │
    ├── workloads/
    │   ├── __init__.py
    │   ├── filtering.py
    │   ├── groupby.py
    │   ├── join.py
    │   └── pipeline.py
    │
    ├── __init__.py
    └── utils.py
```