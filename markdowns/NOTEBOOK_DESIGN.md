# FULL NOTEBOOK DESIGN — END-TO-END (FINAL)

This document defines the **exact purpose, logic, and analysis flow** for every notebook in the project.

---

# 0. GLOBAL RULES (APPLY TO ALL NOTEBOOKS)

* Do NOT run benchmarks
* Only read from:

  * `results/raw/`
  * `data/benchmark_real/`
  * `data/benchmark_syn/`
* Always filter `status == "ok"`
* Always aggregate multiple runs (mean, std)
* Never mix tracks (logical vs physical)

---

# 1. 00_setup_and_verify.ipynb

## Purpose

Sanity check environment and data integrity.

## Analysis

* Load a few result CSV files
* Inspect schema consistency
* Check:

  * missing values
  * unexpected dataset_size
  * framework/workload labels

## Output

* Data preview
* Summary statistics

## Insight

Ensure pipeline is correct before analysis begins.

---

# 2. data_prep/

---

## 01a_verify_real_data.ipynb

## Purpose

Understand characteristics of real dataset (Amazon Reviews).

## Analysis

* Text length distribution
* Rating distribution
* User/item frequency (long-tail / Zipf behavior)
* Cardinality of keys

## Questions

* Is data skewed?
* Is distribution heavy-tailed?

## Insight

Real data structure informs synthetic generation.

---

## 01b_calibrate_synthetic.ipynb

## Purpose

Fit parameters for synthetic data generation.

## Analysis

* Estimate:

  * text length distribution (log-normal)
  * user frequency (Zipf)
  * categorical distributions

## Key Principle

* Calibration done ONCE
* Parameters must remain fixed

## Output

* configuration parameters for generator

---

## 01c_validate_synthetic.ipynb

## Purpose

Validate synthetic data resembles real data.

## Analysis

* Compare:

  * histograms (real vs synthetic)
  * quantiles (p50, p90, p99)
  * entropy / distribution similarity

## Questions

* Are distributions aligned?
* Are differences acceptable?

## Insight

Synthetic must preserve statistical properties of real data.

---

# 3. benchmarks/

---

## 02_filter_benchmark.ipynb

## Research Question

How efficiently do frameworks handle simple filtering?

---

## Analysis Logic

* Compare execution time across frameworks
* Observe scaling (1M → 50M)
* Analyze throughput stability

---

## Key Insight

* Pure scan performance
* CPU vs IO bottleneck behavior

---

## 03_groupby_benchmark.ipynb

## Research Question

How do frameworks perform under aggregation workloads?

---

## Analysis Logic

* Time vs dataset size
* Memory usage growth
* Throughput under aggregation

---

## Key Insight

* Memory pressure behavior
* Aggregation efficiency

---

## 04_join_benchmark.ipynb

## Research Question

How do frameworks scale for join operations?

---

## Analysis Logic

* Time vs dataset size
* Throughput degradation

---

## Key Insight

* Join complexity dominates performance
* Shuffle / hash cost differences

---

## 05_pipeline_benchmark.ipynb

## Research Question

How do frameworks perform on realistic multi-step pipelines?

---

## Analysis Logic

* Compare pipeline vs individual workloads
* Identify cost amplification

---

## Key Insight

* Combined workload effects
* Benefits of lazy execution

---

# 4. analysis/

---

## 06a_logical_scaling.ipynb

## Research Question

What is the algorithmic scalability of each framework?

---

## Analysis Logic

* Log-log plot (rows vs time)
* Estimate scaling slope
* Compute speedup vs baseline (Pandas)

---

## Key Insight

* Complexity class (linear vs super-linear)
* Relative scalability

---

## 06b_physical_scaling.ipynb

## Research Question

How do frameworks behave under memory constraints?

---

## Analysis Logic

### Main (valid workloads)

* Filter + GroupBy only
* Time vs dataset size (GB)
* Memory vs dataset size

---

### Stress Analysis

* Join + Pipeline
* Analyze:

  * slowdown
  * system limits
  * failure patterns

---

## Key Insight

* Out-of-core capability
* Memory bottlenecks
* System-level behavior

---

## 07_real_vs_synthetic.ipynb

## Research Question

Does synthetic data reflect real-world performance?

---

## Analysis Logic

* Compare runtime patterns
* Compare ranking of frameworks

---

## Key Insight

* Validity of synthetic benchmark
* Consistency of conclusions

---

## 08_lazy_vs_eager_polars.ipynb

## Research Question

Does lazy execution improve performance?

---

## Analysis Logic

* Compare lazy vs eager
* Compute speedup
* Analyze workload sensitivity

---

## Key Insight

* Query optimization benefits
* When lazy execution matters

---

# 5. report/

---

## 09_final_report.ipynb

## Purpose

Synthesize all findings into a coherent report.

---

## Structure

### 1. Introduction

* Motivation
* Problem definition

---

### 2. Dataset

* Real vs synthetic
* Data characteristics

---

### 3. Methodology

* Benchmark design
* Tracks and workloads
* Metrics

---

### 4. Results — Logical Scaling

* Insights from 06a

---

### 5. Results — Physical Scaling

* Insights from 06b

---

### 6. Validation

* Real vs synthetic comparison

---

### 7. Discussion

* Trade-offs between frameworks
* Strengths and limitations
* Practical recommendations

---

### 8. Conclusion

* Final takeaways:

  * Pandas → small-scale
  * Polars → high-performance single machine
  * Dask → large-scale / distributed

---

# FINAL PRINCIPLE

Each notebook must:

* Answer a clear research question
* Use correct subset of data
* Provide interpretable insights

Notebooks are not for plotting — they are for reasoning.
