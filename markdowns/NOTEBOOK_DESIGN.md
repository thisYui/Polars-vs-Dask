# FULL NOTEBOOK DESIGN — BENCHMARK-ALIGNED VERSION

This document defines the purpose, scope, and analysis logic for every notebook in the project.

The notebook structure is aligned with the revised benchmark design. The benchmark is organized around four independent benchmark groups:

1. Benchmark 1 — Real Data Row Scaling
2. Benchmark 2 — Physical Scaling by Dataset Size
3. Benchmark 3 — Synthetic Stress Benchmark
4. Benchmark 4 — Real Data OS Comparison: Windows vs Linux

This structure keeps each notebook focused on one research question and avoids mixing variables.

---

# 0. Global Rules

These rules apply to all notebooks from `02` onward.

## Execution Rule

Notebooks must not run benchmark workloads directly.

Benchmark execution is handled separately by scripts such as:

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

Recommended dataset/result naming:

```text
data/benchmark_real/1M/
data/benchmark_real/10M/
data/benchmark_real/50M/

data/benchmark_syn/size_1GB/
data/benchmark_syn/size_5GB/
data/benchmark_syn/size_20GB/
data/benchmark_syn/size_over_ram/

data/benchmark_syn/stress_100M_real_like/
data/benchmark_syn/stress_10M_heavy_skew/
data/benchmark_syn/stress_10M_high_unique_id/

results/raw/windows/
results/raw/linux/
```

## Result Filtering

Always filter benchmark results with:

```python
df_ok = df[df["status"] == "ok"]
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

## Benchmark Separation Rule

Never mix the four benchmark groups in the same analysis table unless the purpose is explicitly comparison across benchmark groups.

Each benchmark controls a different primary variable:

| Benchmark | Primary Variable | Dataset Type | Controlled Variables |
|---|---|---|---|
| Benchmark 1 — Real Row Scaling | Rows | Real only | Same schema, same workload, same environment |
| Benchmark 2 — Physical Scaling | Dataset size / memory footprint | Synthetic only | Fixed synthetic distribution |
| Benchmark 3 — Synthetic Stress | Synthetic stress condition | Synthetic only | Fixed row count where specified, controlled generator variants |
| Benchmark 4 — Real OS Comparison | Operating system | Real only | Same real datasets, same scripts |

---

# 1. `00_setup_and_verify.ipynb`

## Purpose

Sanity-check the project environment, paths, datasets, benchmark result files, and OS metadata before analysis begins.

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
  - benchmark_group
  - os
  - status
  - execution time
  - memory usage
- Check available dataset folders.
- Check schema consistency between real and synthetic datasets.
- Check whether both Windows and Linux result folders exist when OS comparison is enabled.

## Output

- Environment summary.
- Available datasets.
- Available benchmark results.
- Missing file warnings.
- Basic schema preview.
- OS/result availability summary.

## Key Insight

This notebook confirms that the pipeline has produced the expected data and result files before deeper analysis starts.

---

# 2. `data_prep/`

---

## 2.1 `01a_verify_real_data.ipynb`

## Purpose

Understand the statistical structure of the real Amazon Reviews dataset.

This notebook profiles the real data and identifies the distributions that synthetic data must preserve.

## Inputs

```text
data/benchmark_real/1M/
```

Optionally sample-check:

```text
data/benchmark_real/10M/
data/benchmark_real/50M/
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

This notebook does not benchmark framework performance. It only validates data quality.

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

The benchmark notebooks are organized by benchmark group instead of workload.

---

## 3.1 `02_real_row_scaling.ipynb`

## Benchmark

Benchmark 1 — Real Data Row Scaling

## Purpose

Analyze how each framework scales on real data as the number of rows increases.

This is the primary benchmark for real-world performance.

## Research Question

How do Pandas, Polars, and Dask scale on real Amazon Reviews data when row count increases from 1M to 10M to 50M?

## Inputs

```text
results/raw/
data/benchmark_real/
```

## Scope

| Parameter | Value |
|---|---|
| Dataset Type | Real only |
| Row Counts | 1M, 10M, 50M |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |
| Primary Variable | Number of rows |

## Analysis

- Load benchmark results for real row-based datasets.
- Filter `status == "ok"`.
- Aggregate repeated runs by:
  - framework
  - workload
  - dataset_size
  - os, if present
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

- Which framework scales best on real data?
- Is scaling close to linear?
- Which workload becomes super-linear?
- Which framework ranking is stable across 1M, 10M, and 50M?
- Does Polars lazy execution provide consistent benefits?

## Output

- Runtime scaling plots.
- Throughput plots.
- Speedup tables.
- Scaling slope estimates.
- Per-workload interpretation.

## Key Insight

This notebook explains real-world algorithmic and execution-model scalability under row growth.

---

## 3.2 `03_physical_scaling.ipynb`

## Benchmark

Benchmark 2 — Physical Scaling by Dataset Size

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

## 3.3 `04_synthetic_stress.ipynb`

## Benchmark

Benchmark 3 — Synthetic Stress Benchmark

## Purpose

Evaluate framework behavior under synthetic stress scenarios that are difficult or impossible to reproduce with real data alone.

This benchmark is not for validating synthetic data against real data. It is for stress testing framework robustness.

## Research Question

How do Pandas, Polars, and Dask behave under extreme synthetic conditions such as very large row count, heavy key skew, and high ID cardinality?

## Inputs

```text
results/raw/
data/benchmark_syn/
```

## Scope

| Scenario | Dataset | Purpose |
|---|---|---|
| Real-like large scale | Synthetic 100M rows, structure similar to real | Test large-scale row processing (only Dask and Polars Lazy and only fitering and groupby not include join and pipeline) |
| Heavy skew | Synthetic 10M rows with severe skewness | Test hot-key/groupby/join sensitivity (Pandas, Polars Eager, Polar Lazy, Dask) |
| High unique ID | Synthetic 10M rows with many unique IDs | Test cardinality pressure and memory overhead (Pandas, Polars Eager, Polar Lazy, Dask) |

| Parameter | Value |
|---|---|
| Dataset Type | Synthetic only |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |
| Primary Variable | Stress condition |

## Analysis

- Load synthetic stress benchmark results.
- Filter `status == "ok"` for successful-run analysis.
- Analyze failed runs separately.
- Aggregate repeated runs by:
  - stress_scenario
  - framework
  - workload
- Compare:
  - runtime
  - peak memory
  - failure rate
  - throughput
  - slowdown relative to real-like 10M or baseline synthetic 10M, if available
- Inspect workload-specific sensitivity:
  - groupby under heavy skew
  - join under high cardinality
  - pipeline under combined pressure

## Questions

- Can each framework complete the 100M real-like synthetic benchmark?
- Which framework is most sensitive to heavy skew?
- Which framework is most affected by high unique ID cardinality?
- Do failures occur because of memory, timeouts, shuffle cost, or unsupported execution patterns?
- Are framework rankings stable under stress, or do they change?

## Output

- Stress scenario summary table.
- Runtime and memory plots by scenario.
- Failure table.
- Framework robustness ranking.
- Workload-specific stress interpretation.

## Key Insight

This notebook identifies failure modes and robustness limits that are not visible in normal real-data scaling.

---

## 3.4 `05_real_os_comparison.ipynb`

## Benchmark

Benchmark 4 — Real Data OS Comparison: Windows vs Linux

## Purpose

Compare real-data benchmark results between Windows and Linux.

This notebook checks environment sensitivity and reproducibility. It should not be mixed into the main real row scaling conclusions unless OS is explicitly treated as a factor.

## Research Question

Are real-data benchmark trends stable across Windows and Linux?

## Inputs

```text
results/raw/windows/
results/raw/linux/
data/benchmark_real/
```

## Scope

| Parameter | Value |
|---|---|
| Dataset Type | Real only |
| Row Counts | 1M, 10M, 50M |
| Operating Systems | Windows, Linux |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |
| Primary Variable | Operating system |

## Design Rule

Windows and Linux runs should use:

- the same real datasets
- the same benchmark scripts
- the same framework versions when possible
- the same number of repeated runs
- the same hardware if possible

If Windows and Linux are run on the same physical machine, this can be treated as an OS comparison.

If Windows and Linux are run on different machines, report it as an environment comparison rather than a pure OS comparison.

## Analysis

- Load real-data benchmark results from Windows and Linux.
- Filter `status == "ok"`.
- Aggregate repeated runs by:
  - os
  - framework
  - workload
  - dataset_size
- Compute:
  - mean runtime
  - runtime standard deviation
  - mean memory usage
  - OS runtime ratio: `windows_time / linux_time`
  - OS memory ratio: `windows_memory / linux_memory`
- Compare framework ranking per OS.
- Compare workload sensitivity per OS.
- Inspect failures separately.

## Questions

- Are framework rankings stable across Windows and Linux?
- Is Linux consistently faster, or only for certain workloads?
- Which workloads are most OS-sensitive?
- Does Dask show stronger OS sensitivity because of multiprocessing or scheduling behavior?
- Are memory usage patterns similar across OS environments?

## Output

- Windows vs Linux runtime comparison.
- OS runtime ratio table.
- OS memory ratio table.
- Ranking consistency table.
- Failure comparison.
- Reproducibility notes.

## Key Insight

This notebook verifies whether the real-data conclusions are portable across operating systems.

---

# 4. `analysis/`

---

## 4.1 `06_workload_breakdown.ipynb`

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

Use accepted results from Benchmark 1 primarily.

Optionally reference:

- Benchmark 2 for memory pressure behavior
- Benchmark 3 for stress behavior
- Benchmark 4 for OS sensitivity

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
- workload sensitivity to skew/cardinality stress, if relevant

## Questions

- Is join more expensive than groupby?
- Does pipeline amplify costs?
- Which framework is best for scan-heavy workloads?
- Which framework is best for aggregation-heavy workloads?
- Which workload exposes memory bottlenecks?
- Which workload is most sensitive to OS differences?

## Output

- Workload cost ranking.
- Framework-by-workload summary.
- Runtime amplification table.
- Discussion-ready insights.

## Key Insight

This notebook translates raw benchmark results into workload-level interpretation.

---

## 4.2 `07_lazy_vs_eager_polars.ipynb`

## Purpose

Analyze whether Polars lazy execution improves performance or memory behavior over eager execution.

## Research Questions

1. Is Polars Lazy consistently faster than Polars Eager?
2. Which workloads benefit most from lazy execution?
3. Does lazy execution reduce peak memory?
4. Are there cases where Eager is faster or uses less memory?
5. Does lazy-vs-eager behavior change as dataset size increases?
6. What is the practical recommendation for choosing Lazy vs Eager?

## Inputs

Main analysis:

```text
results/raw/polars_real_results.csv
results/raw/polars_eager_real_results.csv
```

Supporting analysis:

```text
results/raw/polars_size_results.csv
results/raw/polars_eager_size_results.csv
results/raw/polars_highuid_skewed_results.csv
results/raw/polars_eager_highuid_skewed_results.csv
results/raw/polars_colab_results.csv
results/raw/polars_eager_colab_results.csv
```

## Analysis Boundary

Main lazy-vs-eager conclusions are based on real-data row-scaling results only.

Physical scaling, synthetic stress, and Colab results are used only as supporting evidence. These groups should not be mixed into the main lazy-vs-eager ranking because they control different primary variables.

## Scope

| Parameter | Value |
|---|---|
| Dataset type | Real data |
| Frameworks | Polars Lazy, Polars Eager |
| Workloads | Filter, GroupBy, Join, Pipeline |
| Dataset sizes | 1M, 10M, 50M |
| Status | ok only |

## Main Metrics

- mean runtime
- runtime standard deviation
- mean peak memory
- memory standard deviation
- mean throughput
- number of successful runs
- lazy speedup
- lazy memory ratio
- dataset-size sensitivity

## Analysis

1. Load Polars Lazy and Eager result files.
2. Validate required columns and result-file inventory.
3. Filter `status == "ok"` for main analysis.
4. Check completeness for:
   - framework group
   - workload
   - dataset size
   - status
5. Aggregate repeated runs by:
   - workload
   - dataset size
   - framework group
6. Compare absolute runtime and memory.
7. Compute lazy speedup:

```text
lazy_speedup = eager_runtime / lazy_runtime
```

Interpretation:

- `lazy_speedup > 1.0` means Lazy is faster.
- `lazy_speedup < 1.0` means Eager is faster.

8. Compute lazy memory ratio:

```text
lazy_memory_ratio = lazy_memory / eager_memory
```

Interpretation:

- `lazy_memory_ratio < 1.0` means Lazy uses less peak memory.
- `lazy_memory_ratio > 1.0` means Lazy uses more peak memory.

9. Analyze dataset-size sensitivity from `1M` to `50M`:

```text
runtime_growth_50M_vs_1M
memory_growth_50M_vs_1M
speedup_change_50M_minus_1M
```

10. Interpret results by workload:
    - Filter: predicate/projection pushdown
    - GroupBy: aggregation optimization
    - Join: eager may be competitive
    - Pipeline: query-plan optimization advantage
11. Use size, stress, and Colab groups only as supporting evidence.
12. Produce a practical recommendation table based on observed speedup and memory ratio.

## Notebook Structure

1. Research Questions
2. Setup and Data Loading
3. Result File Inventory and Analysis Boundary
4. Analysis Scope
5. Data Completeness and Validity Check
6. Lazy vs Eager Summary Table
7. Runtime Comparison: Lazy vs Eager
8. Lazy Speedup over Eager
9. Memory Comparison: Lazy vs Eager
10. Dataset-Size Sensitivity
11. Workload-Specific Interpretation
12. Supporting Evidence: Size, Stress, and Colab
13. Practical Recommendation: When to Use Lazy vs Eager
14. Final Conclusion

## Output

- Lazy vs eager runtime plots.
- Runtime vs dataset size plots by workload.
- Speedup table by workload and dataset size.
- Mean speedup table by workload.
- Memory ratio table.
- Dataset-size sensitivity table.
- Workload-specific interpretation.
- Recommendation on Polars execution mode.

## Key Insight

Lazy execution matters most when query optimization, projection pushdown, predicate pushdown, or whole-plan optimization can reduce work or avoid unnecessary intermediate materialization. The notebook should still report cases where Eager is faster or uses less memory, because this is an end-to-end benchmark that includes scan/read, optimization overhead, memory allocation, and final materialization.

---

# 5. `report/`

---

## 5.1 `09_final_report.ipynb`

## Purpose

Synthesize all findings into a coherent final report.

This notebook should not introduce new experiments. It should summarize and explain results from previous notebooks.

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
- Four-benchmark structure:
  - Real Data Row Scaling
  - Physical Scaling by Dataset Size
  - Synthetic Stress Benchmark
  - Real Data OS Comparison: Windows vs Linux
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
  - OS runtime ratio

### 4. Results — Real Data Row Scaling

Summarize `02_real_row_scaling.ipynb`.

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

### 6. Results — Synthetic Stress Benchmark

Summarize `04_synthetic_stress.ipynb`.

Discuss:

- 100M real-like synthetic scalability
- 10M heavy skew behavior
- 10M high unique ID behavior
- robustness and failure modes

### 7. Validation — Synthetic Data Quality

Summarize `01c_validate_synthetic.ipynb`.

Discuss:

- data distribution similarity
- limitations of synthetic data
- whether synthetic is acceptable for physical and stress benchmarks

### 8. Results — Windows vs Linux Real Data Comparison

Summarize `05_real_os_comparison.ipynb`.

Discuss:

- runtime ratio
- memory ratio
- ranking consistency
- OS-specific limitations
- whether real-data conclusions are reproducible across OS

### 9. Workload Discussion

Summarize `06_workload_breakdown.ipynb`.

Discuss:

- scan-heavy workloads
- aggregation-heavy workloads
- join-heavy workloads
- pipeline amplification

### 10. Polars Lazy vs Eager

Summarize `07_lazy_vs_eager_polars.ipynb`.

Discuss:

- when lazy helps
- when eager is sufficient
- practical recommendation

### 11. Limitations

Discuss:

- single-machine environment
- RAM constraint
- OS/hardware differences if Windows and Linux are not run on the same machine
- synthetic data approximation
- dataset-specific conclusions
- framework configuration sensitivity

### 12. Conclusion

Final takeaways:

- Pandas is suitable for small-scale workloads and simple analysis.
- Polars is strong for high-performance single-machine analytics.
- Dask is useful for larger-than-memory and partitioned workloads, but overhead matters.
- Synthetic stress benchmarks reveal robustness limits not visible in normal real-data scaling.
- Windows vs Linux comparison helps validate reproducibility and deployment sensitivity.

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
│   ├── 02_real_row_scaling.ipynb
│   ├── 03_physical_scaling.ipynb
│   ├── 04_synthetic_stress.ipynb
│   └── 05_real_os_comparison.ipynb
│
├── analysis/
│   ├── 06_workload_breakdown.ipynb
│   └── 07_lazy_vs_eager_polars.ipynb
│
└── report/
    └── 08_final_report.ipynb
```

---

# 7. Final Principle

Each notebook must answer one clear research question.

Notebooks are not for running benchmarks. They are for reasoning over benchmark results.

The project should avoid asking:

```text
Which plot should we draw?
```

and instead ask:

```text
What claim does this notebook support?
```
