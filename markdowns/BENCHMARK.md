# Benchmark Objectives and Methodology

## 1. Overall Benchmark Objective

The objective of this benchmark is to evaluate the practical performance of three dataframe processing frameworks:

- Pandas
- Polars
- Dask

The benchmark focuses on common analytical workloads over parquet datasets:

- Filter
- GroupBy
- Join
- Multi-step Pipeline

The benchmark is designed as an **end-to-end dataframe analytics benchmark**, not as a pure isolated engine benchmark.

In practical usage, users do not only care about the raw speed of an internal execution engine. They care about whether a framework can:

- read the dataset successfully,
- execute the workload,
- handle memory pressure,
- complete within a reasonable time,
- materialize or produce a usable result,
- and remain stable as data size grows.

Therefore, this benchmark measures the full execution path:

```text
parquet input
-> dataframe scan/read
-> query/operator execution
-> framework scheduling or optimization
-> memory allocation
-> final result materialization
-> usable output
```

This makes the benchmark representative of real-world dataframe usage in Python.

---

## 2. What This Benchmark Measures

This benchmark measures **practical end-to-end workload performance**.

The measured runtime includes:

- parquet file scanning,
- parquet decoding,
- column projection and predicate pushdown where supported,
- eager or lazy execution,
- parallel execution where supported,
- Dask scheduler overhead,
- Polars query optimization overhead,
- Pandas eager execution cost,
- memory allocation,
- final result materialization,
- and system-level effects such as OS page cache, swap, pagefile behavior, and memory pressure.

The benchmark therefore evaluates the complete behavior of each framework, rather than only measuring one isolated operator.

This is intentional because real analytical workloads are affected by both engine-level performance and system-level execution behavior.

---

## 3. What This Benchmark Does Not Measure

This benchmark does **not** attempt to measure pure engine execution time in isolation.

It does not fully isolate:

- CPU-only operator execution,
- parquet I/O cost,
- parquet decoding cost,
- memory allocation cost,
- scheduler overhead,
- OS cache effects,
- swap/pagefile behavior,
- or final result materialization cost.

For example, a filter workload does not only measure the cost of evaluating a boolean predicate. It may also include the cost of reading parquet files, decoding columns, allocating the filtered output, and returning a DataFrame.

Similarly, a join workload does not only measure the theoretical join algorithm. It also includes data loading, key representation, memory allocation, partition handling, and framework-specific execution strategy.

Because of this, benchmark results should not be interpreted as pure engine speed. They should be interpreted as practical dataframe workload performance.

---

## 4. Why End-to-End Benchmarking Is Still Valuable

Although this benchmark does not isolate pure engine time, it has strong practical value.

In real-world data analysis, users typically write code such as:

```python
df = pd.read_parquet(path)
result = df[df["rating"] >= threshold]
```

or:

```python
result = (
    pl.scan_parquet(path)
    .filter(pl.col("rating") >= threshold)
    .collect()
)
```

or:

```python
result = ddf.groupby(key).agg(...).compute()
```

These workflows include reading data, executing operations, and producing a result that can be used in Python.

Therefore, measuring end-to-end execution answers a practical question:

> Given the same dataset and the same analytical task, which framework completes the workload faster and more reliably?

This is often more useful than measuring a theoretical operator in isolation.

---

## 5. Important Methodological Clarification

A dataset can be larger than physical RAM and still be processed successfully.

This is because frameworks such as Dask and Polars may process data using:

- lazy execution,
- partitioned execution,
- streaming execution,
- column pushdown,
- batch processing,
- or out-of-core execution strategies.

However, this does not mean that the entire dataset is loaded into memory at once.

In many workloads, only part of the dataset is active in memory at a given time. The final result may also be much smaller than the input dataset.

For example:

```text
Input dataset on disk: 30 GB
Available RAM: 16 GB
GroupBy output: 50 MB
```

This workload can succeed even though the input dataset is larger than RAM.

Therefore, this benchmark should be described as:

> a larger-than-memory end-to-end dataframe analytics benchmark

rather than:

> a benchmark where the full dataset is held in memory.

---

## 6. Final Result Materialization

Some workloads in this benchmark return a complete DataFrame result.

For example:

- Pandas workloads usually read data eagerly and return a Pandas DataFrame.
- Dask workloads often use `.compute()` to materialize the final result.
- Polars lazy workloads often use `.collect()` to materialize the final result.

This means that the benchmark includes the cost of final result materialization.

This is important because final result size can strongly affect runtime and memory usage.

For example:

- A GroupBy workload may produce a small result and therefore remain stable.
- A Filter workload may produce a large result if many rows satisfy the condition.
- A Join workload may produce a result close in size to the input table.
- A Pipeline workload may reduce or expand data depending on operation order.

Therefore, workloads with large output sizes may be affected by memory pressure, pagefile usage, swap behavior, or runtime termination in constrained environments.

This behavior is not a benchmark bug. It is part of practical end-to-end dataframe execution.

---

## 7. Windows vs Linux / Colab Behavior

Benchmark results may differ across execution environments.

A workload that succeeds on Windows may fail on Linux or Google Colab because of differences in:

- available RAM,
- pagefile or swap behavior,
- OS memory management,
- runtime resource limits,
- filesystem performance,
- multiprocessing behavior,
- temporary file handling,
- and runtime termination policies.

On Windows, a workload may survive memory pressure because the system can use the pagefile. This can make the workload slower but still allow it to complete.

On Colab or Linux, the same workload may be killed when memory usage exceeds the runtime limit. This is especially common when a workload materializes a large result using `.compute()` or `.collect()`.

Therefore, if the same code runs on Windows but is killed on Colab, this does not necessarily mean the benchmark code is invalid. It indicates that the workload is sensitive to memory limits and environment-specific behavior.

This is why OS or environment comparison should be reported separately from the main benchmark results.

---

## 8. Benchmark Design Principles

The benchmark design follows four main principles.

### 8.1 Control One Main Variable at a Time

Each benchmark group should vary only one primary factor.

Examples:

- row count,
- physical dataset size,
- synthetic stress condition,
- operating system or execution environment.

This avoids mixing multiple effects in the same comparison.

---

### 8.2 Separate Logical and Physical Scaling

Row count and physical dataset size are not equivalent.

The same number of rows can produce different physical sizes depending on:

- string length,
- text distribution,
- compression ratio,
- column cardinality,
- null ratio,
- and schema characteristics.

Therefore, logical row scaling and physical size scaling must be evaluated separately.

---

### 8.3 Use Real Data for the Main Benchmark

The main benchmark uses real data because it reflects actual data distribution.

Real data provides realistic:

- text lengths,
- key distributions,
- cardinality,
- missing values,
- rating distributions,
- and compression behavior.

This avoids introducing synthetic artifacts into the primary comparison.

---

### 8.4 Use Synthetic Data Only for Controlled Stress Tests

Synthetic data is useful when real data cannot provide a desired stress condition.

Synthetic data should be used for explicit purposes such as:

- very large scale,
- heavy key skew,
- high-cardinality identifiers,
- or controlled physical size.

Synthetic data should not replace real data in the main benchmark.

---

# 9. Final Benchmark Structure

The benchmark is divided into four independent benchmark groups:

1. Benchmark 1 — Real Data Row Scaling
2. Benchmark 2 — Size-Based / Physical Scaling
3. Benchmark 3 — Synthetic Stress Benchmarks
4. Benchmark 4 — Real Data OS / Environment Comparison

Each benchmark group has a separate objective and should be interpreted independently.

---

# 9.1 Benchmark 1 — Real Data Row Scaling

## Objective

Evaluate framework performance as the number of rows increases on real data.

This is the primary benchmark because it measures performance on actual data distribution rather than generated data.

The benchmark answers:

- How does each framework scale as real row count increases?
- Which framework performs best on realistic data?
- How do eager execution, lazy execution, and parallel execution behave as real data grows?
- How stable is each framework when handling increasingly large real datasets?

---

## Configuration

| Parameter | Value |
|---|---|
| Dataset type | Real data |
| Row counts | 1M, 10M, 50M |
| Physical size | Recorded but not controlled |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |
| Output mode | Materialized end-to-end result |

---

## Total Runs

For each workload:

```text
3 row counts x 3 frameworks = 9 runs
```

For all four workloads:

```text
9 runs x 4 workloads = 36 runs
```

---

## Interpretation

Benchmark 1 should be interpreted as the main real-world performance comparison.

It measures practical end-to-end performance on real data as row count increases.

However, it should not be used to make strict claims about memory limits because physical dataset size is not controlled.

For example, a 10M-row dataset and a 50M-row dataset may differ not only in row count but also in compression ratio, string length distribution, and physical file size.

---

# 9.2 Benchmark 2 — Size-Based / Physical Scaling

## Objective

Evaluate framework behavior under increasing physical dataset size and memory pressure.

This benchmark focuses on system-level behavior such as:

- memory usage,
- memory pressure,
- spilling,
- swap or pagefile sensitivity,
- out-of-core execution,
- runtime stability,
- and failure points near or beyond RAM limits.

This benchmark answers:

- How does each framework behave as physical dataset size increases?
- Which framework handles memory pressure best?
- Which framework can continue running when data exceeds RAM?
- Which workloads fail first under memory pressure?
- Does performance degrade gradually or fail abruptly?

---

## Configuration

| Parameter | Value |
|---|---|
| Scaling variable | Physical dataset size |
| Size targets | 1GB, 5GB, 20GB, >RAM |
| Row count | Variable |
| Dataset type | Synthetic or prepared size-controlled dataset |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Representative workload subset |
| Output mode | Materialized result or sink-to-disk, depending on experiment |

---

## Total Runs

For each selected workload:

```text
4 size targets x 3 frameworks = 12 runs
```

---

## Interpretation

Benchmark 2 should be interpreted as a physical-size and memory-pressure benchmark.

It should not be used to make row-scaling claims because row count is not the controlled variable.

This benchmark is especially useful for identifying:

- out-of-memory behavior,
- swap/pagefile effects,
- framework stability,
- and larger-than-memory feasibility.

---

## Notes on Larger-Than-RAM Execution

If a dataset is larger than RAM but the workload succeeds, this does not imply that the full dataset was held in memory.

It means the framework was able to process the workload using partitioned, streaming, lazy, or out-of-core behavior.

If the workload fails, the failure may occur because:

- the input working set is too large,
- the final result is too large,
- the join or groupby hash table is too large,
- memory fragmentation occurs,
- or the runtime kills the process under memory pressure.

---

# 9.3 Benchmark 3 — Synthetic Stress Benchmarks

## Objective

Evaluate framework robustness under controlled synthetic stress conditions that are difficult or impossible to isolate using real data alone.

This benchmark is not intended to replace real-data benchmarking. It is used to test edge cases and stress scenarios.

Synthetic stress tests answer questions such as:

- How does each framework behave at very large row counts?
- How does key skew affect groupby and join performance?
- How does high cardinality affect memory usage?
- Which framework is more robust under difficult data distributions?
- Which execution strategies fail or degrade under stress?

---

## Synthetic Test Cases

| Test Case | Dataset Size | Synthetic Structure | Purpose |
|---|---:|---|---|
| Stress Scale | 100M rows | Same schema and broad structure as real data | Test large-scale processing beyond available real-data scale (only Dask and Polars Lazy) |
| Heavy Skewness | 10M rows | Strongly skewed key distribution | Test GroupBy, Join, partitioning, and hot-key behavior (Pandas, Polars Eager, Polar Lazy, Dask) |
| High Unique IDs | 10M rows | Many unique IDs / high cardinality | Test memory pressure, hash-table size, join cost, and cardinality-sensitive operations  (Pandas, Polars Eager, Polar Lazy, Dask) |

---

## Configuration

| Parameter | Value |
|---|---|
| Dataset type | Synthetic only |
| Test cases | 100M real-like, 10M heavy skewness, 10M high unique IDs |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline, with focus on stress-sensitive workloads |
| Physical size | Recorded, not controlled unless required by test case |
| Output mode | Materialized result or controlled output depending on stress goal |


---

## 9.3.1 Stress Scale — 100M Rows, Real-Like Structure

This dataset should preserve the same schema and broad statistical structure as real data.

Purpose:

- test large-scale execution,
- evaluate whether frameworks can process 100M rows,
- expose memory, scheduling, and execution bottlenecks.

Requirements:

- same columns as real data,
- similar column types,
- similar text-length behavior,
- similar rating/category distribution,
- similar missing-value behavior, if applicable.

This test should not intentionally introduce extreme skew or extreme cardinality. Its purpose is scale.

---

## 9.3.2 Heavy Skewness — 10M Rows

This dataset intentionally introduces strong skewness in key columns such as user ID, product ID, business ID, or join/group keys.

Purpose:

- stress GroupBy performance,
- stress Join performance,
- test hot-key behavior,
- test partition imbalance in distributed execution,
- evaluate whether Dask suffers from uneven partitions.

Expected behavior to observe:

- longer tail latency,
- memory concentration on hot partitions,
- degraded parallel efficiency,
- possible spill or failure in skewed joins/groupbys.

---

## 9.3.3 High Unique IDs — 10M Rows

This dataset intentionally increases ID cardinality.

Purpose:

- stress hash-table construction,
- stress join-key cardinality,
- test memory overhead from many unique strings or IDs,
- evaluate dictionary encoding and categorical optimization.

Expected behavior to observe:

- increased memory usage,
- slower groupby or join due to larger hash maps,
- different behavior between string IDs and encoded categorical IDs.

---

# 9.4 Benchmark 4 — Real Data OS / Environment Comparison

## Objective

Evaluate whether real-data benchmark results are consistent across operating systems or execution environments.

This benchmark checks the impact of environment-level differences on real-data workloads, especially:

- file I/O,
- memory allocation,
- multiprocessing behavior,
- disk cache,
- pagefile/swap behavior,
- temporary files,
- spill behavior,
- and runtime termination under memory pressure.

---

## Configuration

| Parameter | Value |
|---|---|
| Dataset type | Real data |
| Row counts | 1M, 10M, 50M |
| Operating systems / environments | Windows, Linux, Colab if applicable |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |
| Output mode | Same as Benchmark 1 unless stated otherwise |

---

## Design Rule

The Windows and Linux runs should use the same real datasets and the same benchmark scripts.

If both OS runs are executed on the same physical machine, the result can be treated as an OS comparison.

If they are executed on different machines, the result should be reported as an environment comparison, not a pure OS comparison.

For example:

```text
Windows local machine vs Linux on same hardware
-> OS comparison

Windows local machine vs Google Colab runtime
-> environment comparison
```

---

## Total Runs

For each workload:

```text
3 row counts x 2 environments x 3 frameworks = 18 runs per workload
```

For four workloads:

```text
18 runs x 4 workloads = 72 runs
```

If more than two environments are included, total runs should be recalculated accordingly.

---

## Interpretation

Benchmark 4 should be reported separately from the main benchmark tracks.

It should not be mixed into Benchmark 1 unless OS or environment is explicitly included as an experimental factor.

This benchmark is useful for evaluating:

- reproducibility,
- deployment sensitivity,
- memory behavior differences,
- and whether results obtained on one environment generalize to another.

---

# 10. Recommended Additional Metrics

Runtime alone is not enough to interpret benchmark behavior.

The benchmark should record additional metrics where possible:

| Metric | Purpose |
|---|---|
| Runtime | Main performance metric |
| Peak RSS / peak memory | Shows maximum memory pressure |
| Dataset size on disk | Captures physical input size |
| Row count | Captures logical scale |
| Output row count | Explains result materialization cost |
| Output size | Explains memory pressure from final result |
| Disk read bytes | Helps interpret I/O-heavy workloads |
| Disk write bytes | Helps interpret sink or spill behavior |
| Success/failure status | Captures stability |
| Failure reason | Distinguishes OOM, timeout, engine error, etc. |
| Environment | Enables reproducibility |
| Framework version | Avoids version-related ambiguity |
| CPU and RAM | Hardware context |
| Storage type | Explains I/O variation |

---

## 11. Why RAM and I/O Should Not Be Subtracted Directly

It may be tempting to estimate pure engine time using a formula such as:

```text
engine_time = total_time - io_time - memory_time
```

This benchmark should avoid that interpretation.

The reason is that total runtime is not a simple linear sum of independent components.

In dataframe engines, the following stages may overlap:

- file reading,
- parquet decoding,
- filtering,
- aggregation,
- memory allocation,
- scheduling,
- and result construction.

For example, Polars and Dask may pipeline or parallelize operations. As a result, I/O time and compute time are not always separable.

Memory usage also cannot be converted into a simple time component. Peak RSS indicates maximum memory usage, but it does not directly measure how much runtime was spent on allocation, copying, garbage collection, or swap.

Therefore, RAM and I/O metrics should be used to **explain** benchmark results, not to subtract from runtime.

---

## 12. Optional Control Benchmarks

To better interpret engine behavior, optional control benchmarks can be added.

These are not replacements for the main benchmark. They are diagnostic benchmarks.

### 12.1 Scan Count Benchmark

Purpose:

- approximate parquet scan and decode baseline,
- measure how fast each framework can read selected columns and count rows,
- provide a baseline for interpreting operator benchmarks.

Example workload:

```text
read parquet
-> select required column
-> count rows
```

---

### 12.2 Filter Count Benchmark

Purpose:

- measure filter execution with minimal output materialization,
- reduce the effect of returning a large filtered DataFrame,
- better approximate filter operator behavior.

Example workload:

```text
read parquet
-> filter rating >= threshold
-> count matching rows
```

---

### 12.3 Join Count Benchmark

Purpose:

- force the framework to execute a join,
- avoid materializing the full joined table,
- reduce memory pressure from large join outputs.

Example workload:

```text
read left table
-> read right table
-> join on key
-> count joined rows
```

---

### 12.4 Sink-to-Disk Benchmark

Purpose:

- evaluate larger-than-memory practical execution,
- avoid holding the full result in Python memory,
- test whether the framework can stream or partition output to parquet.

Example workload:

```text
read parquet
-> execute workload
-> write result to parquet
```

This is especially useful for Colab or Linux environments where materializing a large result may cause the runtime to be killed.

---

## 13. Synthetic Data Generation Rules

Synthetic generation must be explicit and controlled.

For synthetic benchmarks, each synthetic test case must have a fixed configuration.

This means:

- no hidden recalibration per run,
- no accidental distribution drift,
- all parameters must be recorded,
- random seeds must be fixed,
- generated datasets must be reproducible.

---

## 14. Separate Synthetic Profiles

Benchmark 3 should use separate synthetic profiles:

| Profile | Purpose |
|---|---|
| `real_like_100m` | Large-scale synthetic data with real-like structure |
| `heavy_skew_10m` | Strong skewness stress test |
| `high_unique_id_10m` | High-cardinality ID stress test |

Each profile should be versioned and documented.

---

## 15. Synthetic Generation Model

The synthetic dataset can be generated using distributions derived from real data:

| Feature | Distribution / Rule | Purpose |
|---|---|---|
| Text length | Log-normal or calibrated empirical distribution | Capture heavy-tailed text length behavior |
| Key frequency | Zipf / power-law | Model skew and hot-key behavior |
| Rating | Empirical categorical distribution | Preserve real-world discrete distribution |
| IDs | Controlled cardinality | Test reuse vs uniqueness |
| Text content | Controlled randomness | Balance variability and memory footprint |

---

# 16. Validation of Synthetic Data

Synthetic validation is performed before running Benchmark 3.

Recommended notebook:

```text
notebooks/data_prep/01c_validate_synthetic.ipynb
```

---

## 16.1 Core Validation Metrics

The following metrics should be checked:

- row count,
- file size / memory size,
- schema compatibility with real data,
- null ratio,
- unique values per key column,
- key frequency distribution,
- text-length distribution,
- rating/category distribution,
- p50, p90, p99 for important numeric/text features.

---

## 16.2 Advanced Validation Metrics

Optional deeper validation:

- KL Divergence,
- Wasserstein Distance,
- skewness,
- kurtosis,
- entropy.

---

## 16.3 Validation by Synthetic Profile

| Profile | Required Validation |
|---|---|
| `real_like_100m` | Must be structurally similar to real data |
| `heavy_skew_10m` | Must show intentionally stronger skew than real data |
| `high_unique_id_10m` | Must show intentionally higher ID cardinality than real data |

---

# 17. Notebook Structure

```text
notebooks/
|
├── 00_setup_and_verify.ipynb
|
├── data_prep/
|   ├── 01a_verify_real_data.ipynb
|   ├── 01b_calibrate_synthetic.ipynb
|   └── 01c_validate_synthetic.ipynb
|
├── benchmarks/
|   ├── 02_filter_benchmark.ipynb
|   ├── 03_groupby_benchmark.ipynb
|   ├── 04_join_benchmark.ipynb
|   └── 05_pipeline_benchmark.ipynb
|
├── analysis/
|   ├── 06a_benchmark_1_real_row_scaling.ipynb
|   ├── 06b_benchmark_2_size_scaling.ipynb
|   ├── 06c_benchmark_3_synthetic_stress.ipynb
|   ├── 07_environment_comparison.ipynb
|   └── 08_lazy_vs_eager_polars.ipynb
|
└── report/
    └── 09_final_report.ipynb
```

---

# 18. Execution Strategy

## Workloads

The benchmark workloads remain consistent:

- Filter
- GroupBy
- Join
- Pipeline

---

## Run Strategy

| Benchmark | Purpose | Dataset | Scope |
|---|---|---|---|
| Benchmark 1 | Main performance comparison | Real 1M, 10M, 50M | Full workload set |
| Benchmark 2 | Physical size / memory pressure | Size-controlled datasets | Representative workload subset |
| Benchmark 3 | Synthetic stress testing | Synthetic edge cases | Stress-sensitive workloads |
| Benchmark 4 | OS / environment comparison | Real data | Same scripts across environments |

---

## Recommended Priority

1. Run Benchmark 1 first as the main real-data baseline.
2. Run Benchmark 2 to understand memory and physical-size behavior.
3. Run Benchmark 3 to test edge cases and failure modes.
4. Run Benchmark 4 only when environment consistency is important.

---

# 19. Interpretation Guidelines

## 19.1 Benchmark 1 Interpretation

Use Benchmark 1 to compare real-world performance across frameworks.

Do not use it to make strict claims about memory limits because physical size is not controlled.

---

## 19.2 Benchmark 2 Interpretation

Use Benchmark 2 to compare system behavior under increasing physical dataset size.

Do not use it to make row-scaling claims because row count is variable.

---

## 19.3 Benchmark 3 Interpretation

Use Benchmark 3 to evaluate robustness under synthetic stress conditions.

Do not generalize synthetic stress results directly to normal real-world performance.

Instead, report them as controlled edge-case behavior.

---

## 19.4 Benchmark 4 Interpretation

Use Benchmark 4 to evaluate reproducibility and environment sensitivity.

Do not report it as a pure OS comparison unless the same physical machine is used.

If hardware differs, report it as an environment comparison.

---

# 20. Key Insights

- The benchmark is an end-to-end dataframe analytics benchmark, not a pure engine benchmark.
- The benchmark has practical value because it reflects how users actually run Pandas, Polars, and Dask.
- Dataset larger than RAM does not mean the full dataset is loaded into RAM.
- Dask and Polars may process data lazily, partition-wise, or in streaming mode.
- `.compute()` and `.collect()` may still materialize the final result and create memory pressure.
- Row scaling, physical size scaling, synthetic stress testing, and environment comparison must be separated.
- RAM and I/O metrics should be used to explain benchmark results, not subtracted directly to estimate pure engine time.
- Optional control benchmarks such as `scan_count`, `filter_count`, `join_count`, and `sink_to_disk` can help diagnose engine behavior more clearly.

---

# 21. Final Conclusion

The revised benchmark design consists of four benchmark groups:

1. **Benchmark 1 — Real Data Row Scaling**  
   Measures end-to-end framework performance on real datasets at 1M, 10M, and 50M rows.

2. **Benchmark 2 — Size-Based / Physical Scaling**  
   Measures behavior under increasing physical dataset size and memory pressure.

3. **Benchmark 3 — Synthetic Stress Benchmarks**  
   Measures robustness under controlled synthetic stress cases, including 100M-row scale, heavy skewness, and high-cardinality IDs.

4. **Benchmark 4 — Real Data OS / Environment Comparison**  
   Measures whether benchmark behavior is consistent across Windows, Linux, Colab, or other execution environments.

This structure keeps the benchmark interpretable, reproducible, and aligned with the intended experimental goals.

The results should be interpreted as **practical end-to-end dataframe workload performance**, not as pure isolated engine execution time.
