# Benchmark Design Revision

## 1. Problem Statement

Initial benchmark design attempted to compare:

- Real vs Synthetic datasets
- Multiple dataset sizes (1M, 10M, 50M)
- Multiple frameworks (Pandas, Polars, Dask)

However, several critical issues were identified:

### 1.1 Dataset Size Mismatch
- Same row count does not imply same memory size
- Synthetic data often significantly larger due to:
  - text length distribution
  - string uniqueness
  - lack of compression/reuse

### 1.2 Distribution Drift in Synthetic Data
- Synthetic generator recalibrated per dataset size
- Observed drift:
  - `user_reuse_prob`: 0.88 → 0.73
  - `text_max_len`: decreasing with size

→ This invalidates benchmark consistency

### 1.3 Combinatorial Explosion
- Full cross comparison leads to excessive runs
- Difficult to interpret results due to multiple changing variables

---

## 2. Design Principles

To address these issues, we adopt the following principles:

### Principle 1 — Control Variables
Each experiment must vary only ONE primary variable:
- rows
- size (GB)
- data type

### Principle 2 — Separate Logical vs Physical Scaling
- Logical scaling → algorithm behavior
- Physical scaling → system/memory behavior

### Principle 3 — Freeze Synthetic Distribution
- Synthetic data must be generated from a fixed configuration
- No recalibration per dataset size

---

## 3. Final Benchmark Structure

The benchmark is divided into **three independent tracks**:

---

# 3.1 Track A — Logical Scaling (Row-based)

### Objective
Evaluate computational performance as dataset grows in number of rows.

### Configuration

| Parameter | Value |
|----------|------|
| Rows | 1M → 10M → 50M |
| Data Types | Real + Synthetic |
| Distribution | Fixed |
| Size (GB) | Not controlled |

### Total Runs

$$ 3 sizes × 2 data types × 3 frameworks = 18 runs per workload $$


### Notes
- This is the **primary benchmark track**
- Focus: execution model (lazy vs eager, parallelism, etc.)

---

# 3.2 Track B — Physical Scaling (Memory-based)

### Objective
Evaluate system behavior under increasing memory pressure.

### Configuration

| Parameter | Value |
|----------|------|
| Dataset | Synthetic only |
| Size | 1GB → 5GB → 20GB → >RAM |
| Rows | Variable |
| Distribution | Fixed |

### Total Runs

$$ 4 sizes × 3 frameworks = 12 runs per workload $$


### Notes
- Real data is excluded due to scalability limitation
- Focus:
  - out-of-core execution
  - spilling
  - memory efficiency

---

# 3.3 Track C — Real vs Synthetic Validation

### Objective
Validate whether synthetic data reflects real-world behavior.

### Configuration

| Parameter | Value |
|----------|------|
| Rows | Fixed (e.g., 10M) |
| Data Types | Real vs Synthetic |
| Distribution | Fixed |
| Size | Not controlled |

### Total Runs
$$ 2 data types × 3 frameworks = 6 runs per workload $$


### Notes
- Used to justify synthetic data usage
- Not for scalability analysis

---
## 4. Synthetic Data Generation

### 4.1 Critical Rule

The synthetic data generator MUST remain invariant with respect to dataset size.

This means:
- No recalibration when scaling from 1M → 10M → 50M → larger
- All statistical parameters (distributions, probabilities, cardinality behavior) must be fixed after initial calibration
- Any observed variation across dataset sizes must originate from sampling effects, not configuration changes

Violation of this rule leads to:
- Distribution drift
- Inconsistent workload characteristics
- Invalid benchmark comparisons

---

### 4.3 Generation Model

The synthetic dataset is constructed using statistically grounded distributions derived from real data:

| Feature | Distribution | Purpose |
|--------|-------------|--------|
| text_len | Log-normal | Capture heavy-tailed text length behavior |
| user_id frequency | Zipf | Model skew and hot-key access patterns |
| rating | Categorical (empirical) | Preserve real-world discrete distribution |
| text content | Controlled randomness | Balance variability and memory footprint |

---

## 5. Validation of Synthetic Data

Validation is performed in:
```
01c_verify_synthetic.ipynb
```


### Core Metrics

The following metrics are used to ensure alignment between real and synthetic data:

- Histogram comparison (shape similarity)
- Quantiles (p50, p90, p99)
- Cardinality (unique values per column)
- Entropy (information density)

### Advanced Metrics (Optional)

For deeper statistical validation:

- KL Divergence (distribution similarity)
- Wasserstein Distance (distribution shift)
- Skewness and Kurtosis (shape characteristics)

---

## 6. Notebook Structure 

### Updated Structure
```
notebooks/
│
├── 00_setup_and_verify.ipynb
│
├── data_prep/
│   ├── 01a_verify_real_data.ipynb
│   ├── 01b_calibrate_synthetic.ipynb      ← NEW (QUAN TRỌNG)
│   └── 01c_validate_synthetic.ipynb       ← rename + nâng cấp
│
├── benchmarks/
│   ├── 02_filter_benchmark.ipynb
│   ├── 03_groupby_benchmark.ipynb
│   ├── 04_join_benchmark.ipynb
│   └── 05_pipeline_benchmark.ipynb
│
├── analysis/
│   ├── 06a_logical_scaling.ipynb          ← NEW (tách ra)
│   ├── 06b_physical_scaling.ipynb         ← NEW (tách ra)
│   ├── 07_real_vs_synthetic.ipynb
│   └── 08_lazy_vs_eager_polars.ipynb
│
└── report/
    └── 09_final_report.ipynb
```


---

## 7. Execution Strategy

### Workloads (consistent across all tracks)

- Filter
- GroupBy
- Join
- Pipeline

### Run Strategy

| Track | Purpose | Scope |
|------|--------|------|
| A | Main performance comparison | Full workload set |
| B | Stress / memory testing | Subset of workloads |
| C | Data validity check | Single representative size |

---

## 8. Key Insights

- Synthetic data is not “fake data”, but a statistical approximation of real-world distributions
- Matching row count alone is insufficient; distributional properties must be preserved
- Benchmark design must isolate variables to avoid confounded results
- Over-RAM scenarios are only achievable via synthetic data and are essential for system-level evaluation

---

## 9. Final Conclusion

The revised benchmark design:

- Eliminates bias caused by distribution drift
- Reduces unnecessary experimental complexity
- Separates logical (algorithmic) and physical (system-level) performance concerns
- Establishes a reproducible and defensible methodology

This approach aligns with research-grade benchmarking standards rather than ad-hoc or exploratory experimentation.