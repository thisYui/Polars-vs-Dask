# FULL NOTEBOOK DESIGN — SCENARIO-BASED VERSION

This document defines the purpose, scope, and analysis logic for every notebook in the project.

The notebook structure is revised from a workload-based design to a scenario-based design.  
The benchmark is organized around three independent tracks:

1. Logical Scaling
2. Physical Scaling
3. Real vs Synthetic Runtime Validation

This structure keeps each notebook focused on one research question and avoids mixing variables.

---

# 0. Global Rules

These rules apply to all notebooks from `02` onward.

## Execution Rule

Notebooks must not run benchmark workloads directly.

Benchmark execution is handled separately by:

```bash
python run_pipeline.py ...
python benchmarks/run_all.py ...
python benchmarks/pandas_run.py ...
python benchmarks/polars_run.py ...
python benchmarks/dask_run.py ...
```

Notebooks only read completed benchmark outputs.

## Allowed Inputs

Notebooks may only read from:

```text
results/raw/
data/benchmark_real/
data/benchmark_syn/
```

## Result Filtering

Always filter benchmark results with:

```python
df = df[df["status"] == "ok"]
```

Failed runs should be analyzed separately only when the notebook explicitly discusses failure behavior.

## Aggregation Rule

Always aggregate repeated runs using:

```text
mean
standard deviation
number of successful runs
```

Do not draw conclusions from a single timed run unless the notebook clearly states that it is exploratory.

## Track Separation Rule

Never mix logical scaling, physical scaling, and real-vs-synthetic validation in the same analysis table unless the purpose is explicitly comparison across tracks.

Each track controls a different primary variable:

| Track | Primary Variable | Controlled Variables |
|---|---|---|
| Logical Scaling | Rows | Distribution fixed |
| Physical Scaling | Memory size | Synthetic only, distribution fixed |
| Real vs Synthetic Runtime | Data type | Rows fixed |

---

# 1. `00_setup_and_verify.ipynb`

## Purpose

Sanity-check the project environment, paths, datasets, and benchmark result files before analysis begins.

## Inputs

```text
results/raw/
data/benchmark_real/
data/benchmark_syn/
```

## Analysis

- Check whether expected directories exist.
- Check whether benchmark result CSV files exist.
- Load a small sample of result files.
- Inspect required columns:
  - framework
  - workload
  - dataset_size
  - data_type
  - status
  - execution time
  - memory usage
- Check available dataset folders.
- Check schema consistency between real and synthetic datasets.

## Output

- Environment summary.
- Available datasets.
- Available benchmark results.
- Missing file warnings.
- Basic schema preview.

## Key Insight

This notebook confirms that the pipeline has produced the expected data and result files before deeper analysis starts.

---

# 2. `data_prep/`

---

## 2.1 `01a_verify_real_data.ipynb`

## Purpose

Understand the statistical structure of the real Amazon Reviews dataset.

This notebook profiles the real 1M sample and identifies the distributions that synthetic data must preserve.

## Inputs

```text
data/benchmark_real/1M/
```

## Analysis

- Inspect schema and dtypes.
- Check missing values.
- Analyze text length distribution:
  - mean
  - median
  - p90
  - p99
  - skewness
  - kurtosis
- Analyze rating distribution.
- Analyze `user_id` frequency.
- Analyze `parent_asin` frequency.
- Compare `product_id` vs `parent_asin` cardinality.
- Estimate key cardinalities.
- Estimate memory footprint.

## Important Key Choice

Use:

```python
ITEM_COL = "parent_asin"
```

`parent_asin` is used as the item key because it represents the parent product level.  
`product_id` is closer to SKU or variant level and causes stronger fragmentation.

## Output

- Real data profiling summary.
- Text length statistics.
- Rating empirical probabilities.
- User/item cardinality.
- Zipf-like frequency behavior.
- Memory footprint estimate.

## Key Insight

The real dataset is skewed and heavy-tailed. Synthetic data must preserve:

- text length long-tail
- rating imbalance
- user frequency skew
- product key skew
- realistic memory footprint

---

## 2.2 `01b_calibrate_synthetic.ipynb`

## Purpose

Fit and freeze parameters for the synthetic data generator.

This notebook converts insights from `01a` into fixed generator parameters.

## Inputs

```text
data/benchmark_real/1M/
```

## Analysis

- Fit text length distribution.
- Estimate log-normal parameters:
  - `mu`
  - `sigma`
- Estimate rating empirical distribution.
- Estimate user ID reuse probability.
- Estimate user cardinality.
- Estimate parent product cardinality.
- Estimate Zipf-like alpha values for reference.
- Estimate memory bytes per row.

## Output

Frozen synthetic generator configuration:

```text
TEXT_LEN_LOGNORMAL_MU
TEXT_LEN_LOGNORMAL_SIGMA
TEXT_MIN_LEN
TEXT_MAX_LEN
ID_POOL_REUSE_PROB
RATING_DISTRIBUTION
N_USERS
N_PRODUCTS
ITEM_KEY_COLUMN
```

## Key Principle

Calibration is done once.

After the parameters are frozen, synthetic data generation must not recalibrate per dataset size.

## Key Insight

Synthetic data should scale by row count or memory target while keeping the same distributional configuration.

---

## 2.3 `01c_validate_synthetic.ipynb`

## Purpose

Validate whether synthetic data resembles real data closely enough for benchmarking.

This notebook does not benchmark framework performance.  
It only validates data quality.

## Inputs

```text
data/benchmark_real/1M/
data/benchmark_syn/1M/
```

or another fixed-size real/synthetic pair.

## Analysis

Compare real vs synthetic on:

- shape
- schema
- memory footprint
- text length quantiles
- helpful vote quantiles
- rating distribution
- cardinality:
  - `user_id`
  - `parent_asin`
  - `product_id`
- histogram similarity
- KL divergence
- Wasserstein distance

## Validation Criteria

Synthetic data does not need to match real data perfectly.

It is acceptable if:

- schema matches
- key cardinality is close
- rating distribution is close
- text length median/p90/p99 are close
- helpful vote remains zero-heavy with a long tail
- memory footprint is within a reasonable range
- advanced metrics pass thresholds

## Output

- Validation tables.
- Distribution plots.
- PASS/WARN/FAIL scorecard.
- Decision: accept synthetic data or revise generator.

## Key Insight

Synthetic data is accepted only if it preserves the statistical properties that affect benchmark behavior.

---

# 3. `benchmarks/`

The benchmark notebooks are now organized by scenario/track instead of workload.

---

## 3.1 `02_logical_scaling.ipynb`

## Track

Track A — Logical Scaling

## Purpose

Analyze how each framework scales as the number of rows increases.

This is the primary benchmark track.

## Research Question

How do Pandas, Polars, and Dask scale when dataset row count increases?

## Inputs

```text
results/raw/
data/benchmark_real/
data/benchmark_syn/
```

## Scope

| Parameter | Value |
|---|---|
| Sizes | 1M, 10M, 50M |
| Data Types | Real + Synthetic |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |
| Primary Variable | Number of rows |

## Analysis

- Load benchmark results for row-based sizes.
- Filter `status == "ok"`.
- Aggregate repeated runs by:
  - framework
  - workload
  - dataset_size
  - data_type
- Compute:
  - mean runtime
  - runtime standard deviation
  - mean memory usage
  - throughput
  - speedup vs Pandas
- Plot runtime vs row count.
- Use log-log plots where appropriate.
- Estimate scaling slope.

## Questions

- Which framework scales best as rows increase?
- Is scaling close to linear?
- Which workload becomes super-linear?
- Does synthetic produce similar scaling trends to real data?
- Does Polars lazy execution provide consistent benefits?

## Output

- Runtime scaling plots.
- Throughput plots.
- Speedup tables.
- Scaling slope estimates.
- Per-workload interpretation.

## Key Insight

This notebook explains algorithmic and execution-model scalability under row growth.

---

## 3.2 `03_physical_scaling.ipynb`

## Track

Track B — Physical Scaling

## Purpose

Analyze framework behavior under increasing memory pressure.

## Research Question

How do frameworks behave when dataset size approaches or exceeds available RAM?

## Inputs

```text
results/raw/
data/benchmark_syn/
```

## Scope

| Parameter | Value |
|---|---|
| Data Type | Synthetic only |
| Sizes | 1GB, 5GB, 20GB, >RAM |
| Frameworks | Pandas, Polars, Dask |
| Primary Variable | Dataset memory size |
| Distribution | Fixed |

## Workload Scope

Main valid workloads:

```text
filter
groupby
```

Stress workloads:

```text
join
pipeline
```

Join and pipeline may be analyzed separately because they can trigger memory limits, shuffle overhead, or failures.

## Analysis

- Load memory-based synthetic benchmark results.
- Filter successful runs for main analysis.
- Separately inspect failed runs.
- Aggregate repeated runs.
- Compare:
  - runtime vs GB size
  - memory usage vs GB size
  - success/failure behavior
  - out-of-core capability
  - slowdown near RAM limit
- Identify frameworks that fail or degrade sharply.

## Questions

- Which framework handles larger-than-RAM data best?
- Does Dask benefit from partitioning?
- Does Polars streaming help under memory pressure?
- Where does Pandas become impractical?
- Which workloads cause memory stress first?

## Output

- Runtime vs memory-size plots.
- Memory usage plots.
- Failure summary table.
- Stress analysis for join and pipeline.
- Practical scale limits.

## Key Insight

This notebook explains system-level scalability, memory pressure, and out-of-core behavior.

---

## 3.3 `04_real_vs_synthetic_runtime.ipynb`

## Track

Track C — Real vs Synthetic Runtime Validation

## Purpose

Validate whether synthetic data produces runtime behavior similar to real data.

This notebook justifies the use of synthetic data for larger-scale benchmark scenarios.

## Research Question

Does synthetic data reflect real-world framework performance patterns?

## Inputs

```text
results/raw/
data/benchmark_real/
data/benchmark_syn/
```

## Scope

| Parameter | Value |
|---|---|
| Rows | Fixed representative size, e.g. 10M |
| Data Types | Real vs Synthetic |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |
| Primary Variable | Data type |

## Analysis

- Select one fixed row count.
- Compare real vs synthetic runtime by workload and framework.
- Compare framework ranking:
  - fastest
  - slowest
  - relative speedup
- Compare memory usage.
- Check whether synthetic preserves performance ordering.

## Questions

- Does synthetic preserve the same framework ranking as real data?
- Are runtime differences acceptable?
- Which workloads are most sensitive to real vs synthetic differences?
- Can synthetic data be trusted for physical scaling?

## Output

- Real vs synthetic runtime comparison.
- Ranking consistency table.
- Memory comparison.
- Workload sensitivity notes.

## Key Insight

Synthetic data is valid for benchmark extension if it preserves framework ranking and workload behavior sufficiently well.

---

# 4. `analysis/`

---

## 4.1 `05_workload_breakdown.ipynb`

## Purpose

Compare workload characteristics across the benchmark.

This notebook summarizes how different workload types affect performance.

## Research Question

Which workloads are most expensive, and which frameworks are best suited to each workload type?

## Inputs

```text
results/raw/
```

## Scope

Use accepted results from Track A primarily.  
Optionally reference Track B stress results.

## Analysis

Compare workloads:

```text
filter
groupby
join
pipeline
```

Analyze:

- average runtime by workload
- relative cost vs filter baseline
- memory usage by workload
- framework ranking by workload
- workload sensitivity to dataset size

## Questions

- Is join more expensive than groupby?
- Does pipeline amplify costs?
- Which framework is best for scan-heavy workloads?
- Which framework is best for aggregation-heavy workloads?
- Which workload exposes memory bottlenecks?

## Output

- Workload cost ranking.
- Framework-by-workload summary.
- Runtime amplification table.
- Discussion-ready insights.

## Key Insight

This notebook translates raw benchmark results into workload-level interpretation.

---

## 4.2 `06_lazy_vs_eager_polars.ipynb`

## Purpose

Analyze whether Polars lazy execution improves performance over eager execution.

## Research Question

When does lazy execution help Polars, and how large is the benefit?

## Inputs

```text
results/raw/
```

## Scope

Compare:

```text
polars_lazy
polars_eager
```

Across:

```text
filter
groupby
join
pipeline
```

where results are available.

## Analysis

- Filter Polars lazy/eager results.
- Aggregate repeated runs.
- Compute speedup:

```text
speedup = eager_time / lazy_time
```

- Compare by workload and dataset size.
- Identify workloads where optimization matters most.

## Questions

- Is lazy consistently faster?
- Which workload benefits most from lazy query optimization?
- Does lazy execution reduce memory pressure?
- Are there cases where eager is comparable or faster?

## Output

- Lazy vs eager runtime plots.
- Speedup table.
- Workload-specific interpretation.
- Recommendation on Polars execution mode.

## Key Insight

Lazy execution matters most when query optimization, projection pushdown, predicate pushdown, or streaming can reduce work.

---

# 5. `report/`

---

## 5.1 `09_final_report.ipynb`

## Purpose

Synthesize all findings into a coherent final report.

This notebook should not introduce new experiments.  
It should summarize and explain results from previous notebooks.

## Structure

### 1. Introduction

- Motivation.
- Why compare Pandas, Polars, and Dask?
- Why large-scale tabular review data?

### 2. Dataset

- Amazon Reviews real dataset.
- Real vs synthetic data.
- Why synthetic data is needed.
- Summary of data validation.

### 3. Methodology

- Benchmark design.
- Three-track structure:
  - Logical Scaling
  - Physical Scaling
  - Real vs Synthetic Runtime Validation
- Workloads:
  - Filter
  - GroupBy
  - Join
  - Pipeline
- Metrics:
  - runtime
  - throughput
  - memory
  - speedup
  - failure behavior

### 4. Results — Logical Scaling

Summarize `02_logical_scaling.ipynb`.

Discuss:

- scaling with rows
- framework ranking
- workload-specific behavior
- speedup vs Pandas

### 5. Results — Physical Scaling

Summarize `03_physical_scaling.ipynb`.

Discuss:

- memory pressure
- larger-than-RAM behavior
- failure patterns
- out-of-core capability

### 6. Validation — Real vs Synthetic

Summarize:

- `01c_validate_synthetic.ipynb`
- `04_real_vs_synthetic_runtime.ipynb`

Discuss:

- data distribution similarity
- runtime behavior similarity
- limitations of synthetic data

### 7. Workload Discussion

Summarize `05_workload_breakdown.ipynb`.

Discuss:

- scan-heavy workloads
- aggregation-heavy workloads
- join-heavy workloads
- pipeline amplification

### 8. Polars Lazy vs Eager

Summarize `06_lazy_vs_eager_polars.ipynb`.

Discuss:

- when lazy helps
- when eager is sufficient
- practical recommendation

### 9. Limitations

Discuss:

- single-machine environment
- 16 GB RAM constraint
- synthetic data approximation
- dataset-specific conclusions
- framework configuration sensitivity

### 10. Conclusion

Final takeaways:

- Pandas is suitable for small-scale workloads and simple analysis.
- Polars is strong for high-performance single-machine analytics.
- Dask is useful for larger-than-memory and partitioned workloads, but overhead matters.
- Synthetic data is valid only after distribution and runtime validation.

---

# 6. Final Notebook List

```text
notebooks/
│
├── 00_setup_and_verify.ipynb
│
├── data_prep/
│   ├── 01a_verify_real_data.ipynb
│   ├── 01b_calibrate_synthetic.ipynb
│   └── 01c_validate_synthetic.ipynb
│
├── benchmarks/
│   ├── 02_logical_scaling.ipynb
│   ├── 03_physical_scaling.ipynb
│   └── 04_real_vs_synthetic_runtime.ipynb
│
├── analysis/
│   ├── 05_workload_breakdown.ipynb
│   └── 06_lazy_vs_eager_polars.ipynb
│
└── report/
    └── 09_final_report.ipynb
```

---

# 7. Final Principle

Each notebook must answer one clear research question.

Notebooks are not for running benchmarks.  
They are for reasoning over benchmark results.

The project should avoid asking:

```text
Which plot should we draw?
```

and instead ask:

```text
What claim does this notebook support?
```
