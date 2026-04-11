# Big Data Benchmark Project

## Pandas vs Polars vs Dask

---

# 1. Objective

Mục tiêu của project là so sánh ba framework xử lý dữ liệu:

* Pandas (baseline)
* Polars
* Dask

Project tập trung vào các yếu tố:

* Performance
* Memory usage
* Scalability
* Lazy vs eager execution
* Khả năng xử lý dữ liệu lớn hơn RAM

---

# 2. Research Questions

Project nhằm trả lời các câu hỏi:

1. Khi dataset tăng kích thước, framework nào scale tốt nhất?
2. Framework nào xử lý được dataset lớn hơn RAM?
3. Framework nào nhanh nhất với groupby?
4. Framework nào tối ưu tốt pipeline nhiều bước?
5. Lazy execution có cải thiện performance không?
6. Columnar engine (Polars) vs row-based (Pandas)?
7. Distributed execution (Dask) có lợi thế khi nào?

---

# 3. Framework Characteristics

## Pandas

Đặc điểm:

* single-threaded
* eager execution
* in-memory
* row-based

Ưu điểm:

* dễ dùng
* phổ biến

Nhược điểm:

* không scale
* dễ hết RAM

---

## Polars

Đặc điểm:

* multi-threaded
* columnar engine
* lazy execution
* SIMD optimized

Ưu điểm:

* rất nhanh trên single machine
* tối ưu CPU

Nhược điểm:

* vẫn bị giới hạn RAM

---

## Dask

Đặc điểm:

* distributed
* out-of-core
* lazy execution
* partition-based

Ưu điểm:

* xử lý dataset lớn hơn RAM
* scale tốt

Nhược điểm:

* overhead scheduling
* chậm hơn với data nhỏ

---

# 4. Dataset

Project sử dụng **Amazon Reviews Dataset**

Dataset:

Amazon Product Reviews

Đặc trưng:

* dữ liệu lớn
* nhiều text
* nhiều user
* phù hợp groupby
* phù hợp join

---

# 4.1 Dataset Columns

Các cột chính:

* review_id
* user_id
* product_id
* rating
* review_text
* review_time
* category
* verified_purchase

---

# 4.2 Dataset Size Strategy

Project sẽ test nhiều kích thước:

| Level  | Rows |
| ------ | ---- |
| small  | 1M   |
| medium | 5M   |
| large  | 10M  |
| xlarge | 50M  |
| huge   | 100M |

---

# 4.3 Dataset Larger Than RAM

Một thí nghiệm quan trọng:

Dataset size > RAM

Ví dụ:

RAM máy: 8GB

Dataset:

* 12GB
* 20GB
* 30GB

Kỳ vọng:

Pandas → crash hoặc MemoryError
Polars → có thể crash
Dask → vẫn chạy được

Mục tiêu:

Chứng minh khả năng xử lý big data thực sự.

---

# 5. Metrics

Các metrics sẽ đo:

## Performance

* execution time
* throughput

## Memory

* peak RAM usage
* memory consumption

## Scalability

* performance theo dataset size

---

# 6. Workloads

Project benchmark 4 workload.

---

# 6.1 Filtering

Filter theo rating

Ví dụ:

rating >= 4

Mục tiêu:

test scan speed

---

# 6.2 GroupBy Aggregation

Group theo product

Tính:

* avg rating
* count review
* sum

Mục tiêu:

test aggregation

---

# 6.3 Join

Join:

review + product metadata

Mục tiêu:

test large join

---

# 6.4 Complex Pipeline

Pipeline:

filter → groupby → join → sort

Mục tiêu:

simulate real analytics

---

# 7. Execution Methodology

Để benchmark công bằng:

## Warmup

Chạy 1 lần trước khi đo

---

## Repeat

Mỗi workload chạy 3 lần

---

## Average

Lấy trung bình

---

## Force Execution

Pandas

eager → không cần

Polars lazy

phải gọi collect()

Dask

phải gọi compute()

---

# 8. Benchmark Strategy

Quy trình benchmark:

1 generate dataset
2 load dataset
3 run workload
4 measure time
5 measure memory
6 save result

---

# 9. Additional Experiments

## Lazy vs Eager

Polars eager vs lazy

---

## IO Benchmark

CSV vs Parquet

---

## Partition Tuning

Dask partition size

---

## Thread Scaling

Polars threads

---

## Dataset Larger Than RAM

thử dataset lớn hơn RAM

---

# 10. Output

Project sẽ output:

## Tables

framework vs time

---

## Graphs

execution time vs size

memory vs size

---

## Analysis

So sánh behavior

---

# 11. Expected Results

Small dataset

Pandas competitive

Medium dataset

Polars fastest

Large dataset

Polars nhanh nhất single machine

Huge dataset > RAM

Dask only solution

---

# 12. Final Goal

Project chứng minh:

Pandas phù hợp small data
Polars phù hợp single machine big data
Dask phù hợp distributed big data
