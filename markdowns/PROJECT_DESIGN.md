# Big Data Benchmark Project

## 1. Objective

Compare Pandas, Polars, and Dask in terms of:

- Execution time
- Memory usage
- Scalability
- Execution model (lazy vs eager)
- Out-of-core capability

---

## 2. Benchmark Structure

The benchmark is divided into three independent tracks:

### Track A — Logical Scaling (Row-based)

- Rows: 1M, 10M, 50M
- Data: Real + Synthetic
- Purpose: evaluate algorithmic scalability

---

### Track B — Physical Scaling (Memory-based)

- Size: 1GB, 5GB, 20GB, >RAM
- Data: Synthetic only
- Purpose: evaluate memory behavior and out-of-core execution

---

### Track C — Real vs Synthetic Validation

- Rows: fixed (e.g., 10M)
- Data: Real vs Synthetic
- Purpose: validate representativeness of synthetic data

---

## 3. Workloads

- Filtering
- GroupBy Aggregation
- Join
- Pipeline (multi-step)

---

## 4. Execution Methodology

- Warmup: 1 run (discarded)
- Repeat: 3 runs
- Metrics:
  - execution time
  - peak memory
  - throughput (rows/sec)

- Force execution:
  - Pandas: eager
  - Polars: `.collect()`
  - Dask: `.compute()`

---

## 5. Dataset Strategy

- Real data: Amazon Reviews
- Synthetic data:
  - fixed distribution (no recalibration)
  - calibrated once from real data

---

## 6. Result Storage

All benchmark outputs are stored in:

results/raw/

File naming convention:

{framework}_{workload}_{track}_{datatype}.csv

---

## 7. Analysis Workflow

- Notebooks DO NOT run benchmarks
- Only load precomputed results
- Perform:
  - aggregation
  - visualization
  - comparative analysis

---

## 8. Key Principles

- Control variables (only one changing factor per track)
- Separate logical vs physical scaling
- Ensure reproducibility