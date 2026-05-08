# Benchmark Design Revision

## 1. Problem Statement

The original benchmark design attempted to compare too many dimensions at once:

- Real vs Synthetic datasets
- Multiple row counts: 1M, 10M, 50M
- Multiple physical sizes: GB-based scaling
- Multiple frameworks: Pandas, Polars, Dask

This creates several issues:

### 1.1 Dataset Size Mismatch

Same row count does not imply same physical memory or disk size.

Synthetic data can be significantly larger than real data because of:

- longer text distribution
- higher string uniqueness
- less compression/reuse
- different cardinality patterns

Therefore, row-based comparison and size-based comparison must be separated.

### 1.2 Distribution Drift in Synthetic Data

Synthetic data should not be recalibrated independently for each dataset size.

Observed drift such as:

- `user_reuse_prob`: 0.88 → 0.73
- `text_max_len`: decreasing with size

invalidates benchmark consistency because the workload characteristics change together with the scale.

### 1.3 Combinatorial Explosion

A full cross-product of:

- real/synthetic
- row count
- physical size
- framework
- workload

creates too many runs and makes the results difficult to interpret.

The revised benchmark therefore separates the experiment into three clearly scoped benchmark groups.

---

## 2. Design Principles

### Principle 1 — Control One Main Variable

Each benchmark should vary only one primary dimension:

- number of rows
- physical dataset size
- synthetic stress condition

### Principle 2 — Separate Logical and Physical Scaling

- Logical scaling evaluates algorithmic behavior as row count increases.
- Physical scaling evaluates memory pressure and system-level behavior as dataset size increases.

### Principle 3 — Use Real Data for Main Row-Based Benchmark

The main benchmark should use real data at 1M, 10M, and 50M rows.

This avoids introducing synthetic-data artifacts into the primary performance comparison.

### Principle 4 — Use Synthetic Data Only for Controlled Stress Tests

Synthetic data is used when real data cannot provide the desired scale or stress condition.

Synthetic benchmarks must have explicit purposes, such as:

- testing very large scale
- testing heavy skewness
- testing high-cardinality identifiers

---

## 3. Final Benchmark Structure

The final benchmark is divided into three independent benchmark groups.

---

# 3.1 Benchmark 1 — Real Data Row Scaling

## Objective

Evaluate framework performance as the number of rows increases on real data.

This is the primary benchmark because it measures behavior on actual data distribution rather than generated data.

## Configuration

| Parameter | Value |
|----------|------|
| Dataset type | Real only |
| Rows | 1M → 10M → 50M |
| Size | Not controlled |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |

## Total Runs

For each workload:

$$ 3\ row\ counts \times 3\ frameworks = 9\ runs $$

For all 4 workloads:

$$ 9\ runs \times 4\ workloads = 36\ runs $$

## Purpose

This benchmark answers:

- How does each framework scale with real row count?
- Which framework performs best on actual data distribution?
- How do lazy execution, eager execution, and parallel execution behave as real data grows?

## Notes

- This benchmark uses real data at 1M, 10M, and 50M rows.
- Physical size is recorded but not controlled.
- Synthetic data is not included in this benchmark to avoid confounding row scaling with generated-data artifacts.

---

# 3.2 Benchmark 2 — Size-Based / Physical Scaling

## Objective

Evaluate system behavior under increasing physical dataset size and memory pressure.

This benchmark focuses on memory usage, spilling, out-of-core execution, and stability near or beyond RAM limits.

## Configuration

| Parameter | Value |
|----------|------|
| Scaling variable | Physical size |
| Size targets | 1GB → 5GB → 20GB → >RAM |
| Rows | Variable |
| Dataset type | Synthetic or prepared size-controlled dataset |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Subset of representative workloads |

## Total Runs

For each selected workload:

$$ 4\ size\ targets \times 3\ frameworks = 12\ runs $$

## Purpose

This benchmark answers:

- How does each framework behave as physical data size increases?
- Which framework handles memory pressure best?
- When does each framework fail, spill, or degrade significantly?
- Which workloads are feasible beyond available RAM?

## Notes

- Row count is not the controlled variable in this benchmark.
- Physical size is the main variable.
- This benchmark is separate from Benchmark 1 because same row count can produce very different physical sizes.
- This benchmark already exists in the notes and should remain as the size-based benchmark.

---

# 3.3 Benchmark 3 — Synthetic Stress Benchmarks

## Objective

Evaluate framework robustness under controlled synthetic stress conditions that are difficult or impossible to isolate using real data alone.

This benchmark is not intended to replace real-data benchmarking. It is used to test edge cases and stress scenarios.

## Synthetic Test Cases

| Test Case | Dataset Size | Synthetic Structure | Purpose |
|----------|--------------|--------------------|---------|
| Stress Scale | 100M rows | Same schema and structure as real data | Test large-scale processing beyond available real-data scale |
| Heavy Skewness | 10M rows | Strongly skewed key distribution | Test GroupBy, Join, partitioning, and hot-key behavior |
| High Unique IDs | 10M rows | Many unique IDs / high cardinality | Test memory pressure, hash-table size, join cost, and cardinality-sensitive operations |

## Configuration

| Parameter | Value |
|----------|------|
| Dataset type | Synthetic only |
| Test cases | 100M real-like, 10M heavy skewness, 10M high unique IDs |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline, with focus on stress-sensitive workloads |
| Size | Recorded, not controlled unless required by test case |

## Total Runs

For each workload:

$$ 3\ synthetic\ cases \times 3\ frameworks = 9\ runs $$

For all 4 workloads:

$$ 9\ runs \times 4\ workloads = 36\ runs $$

## Test Case Details

### 3.3.1 Stress Scale — 100M Rows, Real-Like Structure

This dataset should preserve the same schema and broad statistical structure as real data.

Purpose:

- test large-scale execution
- evaluate whether frameworks can process 100M rows
- expose memory, scheduling, and execution bottlenecks

Requirements:

- same columns as real data
- similar column types
- similar text-length behavior
- similar rating/category distribution
- similar missing-value behavior, if applicable

This test should not intentionally introduce extreme skew or extreme cardinality. Its purpose is scale.

---

### 3.3.2 Heavy Skewness — 10M Rows

This dataset intentionally introduces strong skewness in key columns such as `user_id`, `business_id`, or other join/group keys.

Purpose:

- stress GroupBy performance
- stress Join performance
- test hot-key behavior
- test partition imbalance in distributed execution
- evaluate whether Dask suffers from uneven partitions

Expected behavior to observe:

- longer tail latency
- memory concentration on hot partitions
- degraded parallel efficiency
- possible spill or failure in skewed joins/groupbys

---

### 3.3.3 High Unique IDs — 10M Rows

This dataset intentionally increases ID cardinality.

Purpose:

- stress hash-table construction
- stress join-key cardinality
- test memory overhead from many unique strings or IDs
- evaluate dictionary encoding and categorical optimization

Expected behavior to observe:

- increased memory usage
- slower groupby/join due to larger hash maps
- different behavior between string IDs and encoded categorical IDs

# 3.4 Benchmark 4 — Real Data OS Comparison: Windows vs Linux

## Objective
Evaluate whether real-data benchmark results are consistent across operating systems.

This benchmark checks the impact of OS-level differences on real-data workloads, especially:
- file I/O
- memory allocation
- multiprocessing behavior
- disk cache
- temporary files and spill behavior

## Configuration

| Parameter | Value |
|----------|------|
| Dataset Type | Real only |
| Row Counts | 1M, 10M, 50M |
| Operating Systems | Windows, Linux |
| Frameworks | Pandas, Polars, Dask |
| Workloads | Filter, GroupBy, Join, Pipeline |

## Design Rule

The Windows and Linux runs should use the same real datasets and the same benchmark scripts.

If both OS runs are executed on the same physical machine, the result can be treated as an OS comparison.

If they are executed on different machines, the result should be reported as an environment comparison, not a pure OS comparison.

## Total Runs

$$
3\ row\ counts \times 2\ operating\ systems \times 3\ frameworks
= 18\ runs\ per\ workload
$$

## Notes

- This benchmark should be reported separately from the main benchmark tracks.
- Do not mix Windows/Linux results into Benchmark 1 unless OS is explicitly included as a factor.
- Use this benchmark to evaluate reproducibility and deployment sensitivity.
- Linux results can also serve as a reference environment for larger-scale or production-like runs.

---


## 4. Synthetic Data Generation Rules

### 4.1 Critical Rule

Synthetic generation must be explicit and controlled.

For Benchmark 3, each synthetic test case must have a fixed configuration.

This means:

- no hidden recalibration per run
- no accidental distribution drift
- all parameters must be recorded
- random seeds must be fixed
- generated datasets must be reproducible

### 4.2 Separate Synthetic Profiles

Benchmark 3 should use separate synthetic profiles:

| Profile | Purpose |
|--------|---------|
| `real_like_100m` | Large-scale synthetic data with real-like structure |
| `heavy_skew_10m` | Strong skewness stress test |
| `high_unique_id_10m` | High-cardinality ID stress test |

Each profile should be versioned and documented.

### 4.3 Generation Model

The synthetic dataset can be generated using distributions derived from real data:

| Feature | Distribution / Rule | Purpose |
|--------|---------------------|---------|
| `text_len` | Log-normal or calibrated empirical distribution | Capture heavy-tailed text length behavior |
| key frequency | Zipf / power-law | Model skew and hot-key behavior |
| rating | Empirical categorical distribution | Preserve real-world discrete distribution |
| IDs | Controlled cardinality | Test reuse vs uniqueness |
| text content | Controlled randomness | Balance variability and memory footprint |

---

## 5. Validation of Synthetic Data

Synthetic validation is performed before running Benchmark 3.

Recommended notebook:

```text
notebooks/data_prep/01c_validate_synthetic.ipynb
```

## Core Metrics

The following metrics should be checked:

- row count
- file size / memory size
- schema compatibility with real data
- null ratio
- unique values per key column
- key frequency distribution
- text-length distribution
- rating/category distribution
- p50, p90, p99 for important numeric/text features

## Advanced Metrics

Optional deeper validation:

- KL Divergence
- Wasserstein Distance
- skewness
- kurtosis
- entropy

## Validation by Synthetic Profile

| Profile | Required Validation |
|--------|---------------------|
| `real_like_100m` | Must be structurally similar to real data |
| `heavy_skew_10m` | Must show intentionally stronger skew than real data |
| `high_unique_id_10m` | Must show intentionally higher ID cardinality than real data |

---

## 6. Notebook Structure

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
│   ├── 02_filter_benchmark.ipynb
│   ├── 03_groupby_benchmark.ipynb
│   ├── 04_join_benchmark.ipynb
│   └── 05_pipeline_benchmark.ipynb
│
├── analysis/
│   ├── 06a_benchmark_1_real_row_scaling.ipynb
│   ├── 06b_benchmark_2_size_scaling.ipynb
│   ├── 06c_benchmark_3_synthetic_stress.ipynb
│   └── 08_lazy_vs_eager_polars.ipynb
│
└── report/
    └── 09_final_report.ipynb
```

---

## 7. Execution Strategy

### Workloads

The benchmark workloads remain consistent:

- Filter
- GroupBy
- Join
- Pipeline

### Run Strategy

| Benchmark | Purpose | Dataset | Scope |
|----------|---------|---------|-------|
| Benchmark 1 | Main performance comparison | Real 1M, 10M, 50M | Full workload set |
| Benchmark 2 | Physical size / memory pressure | Size-controlled datasets | Representative workload subset |
| Benchmark 3 | Synthetic stress testing | Synthetic edge cases | Stress-sensitive workloads |

### Recommended Priority

1. Run Benchmark 1 first as the main real-data baseline.
2. Run Benchmark 2 to understand memory and physical-size behavior.
3. Run Benchmark 3 to test edge cases and failure modes.

---

## 8. Interpretation Guidelines

### Benchmark 1 Interpretation

Use Benchmark 1 to compare real-world performance across frameworks.

Do not use it to make claims about memory limits because physical size is not controlled.

### Benchmark 2 Interpretation

Use Benchmark 2 to compare system behavior under increasing dataset size.

Do not use it to make row-scaling claims because row count is variable.

### Benchmark 3 Interpretation

Use Benchmark 3 to evaluate robustness under synthetic stress conditions.

Do not generalize synthetic stress results directly to normal real-world performance.

Instead, report them as controlled edge-case behavior.

---

## 9. Key Insights

- Benchmark 1 should use real data at 1M, 10M, and 50M rows.
- Benchmark 2 should remain size-based, as already noted.
- Benchmark 3 should be synthetic and focused on stress cases.
- Synthetic data is useful for controlled stress tests, not as a replacement for real data.
- Row scaling, size scaling, and stress testing must be separated to avoid confounded results.

---

## 10. Final Conclusion

The revised benchmark design consists of:

1. **Benchmark 1 — Real Data Row Scaling**  
   Real datasets at 1M, 10M, and 50M rows.

2. **Benchmark 2 — Size-Based / Physical Scaling**  
   Dataset size scaling to evaluate memory pressure and system behavior.

3. **Benchmark 3 — Synthetic Stress Benchmarks**  
   Synthetic datasets for:
   - 100M rows with real-like structure
   - 10M rows with heavy skewness
   - 10M rows with many unique IDs

This structure keeps the benchmark interpretable, reproducible, and aligned with the intended experimental goals.
