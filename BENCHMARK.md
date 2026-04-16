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
