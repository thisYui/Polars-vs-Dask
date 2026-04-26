# Dataset Download Guide

## Selected Dataset

We use the **Clothing_Shoes_and_Jewelry** category to ensure the dataset contains **more than 50 million rows**, suitable for large-scale benchmarking.

---

## Download Methods

There are two ways to obtain the dataset:

---

### 1. Using the Pipeline

The dataset can be downloaded through the project pipeline.

#### Advantages
- Allows controlling the number of rows (e.g., 1M, 10M, 50M)
- Automatically saved in compressed `.jsonl.gz` format
- Reduces storage usage
- Fully integrated with preprocessing steps

#### Disadvantages
- Slower due to streaming
- Data must be downloaded before inspection

---

### 2. Direct Download from Hugging Face

Dataset source:  
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/review_categories

Steps:
1. Open the link above  
2. Select `Clothing_Shoes_and_Jewelry.jsonl`  
3. Download the file directly  

#### Advantages
- Faster download via HTTP
- Immediate access to raw data

#### Disadvantages
- File is uncompressed (`.jsonl`)
- Much larger size (can be tens of GB)
- No control over dataset size
- Requires manual processing

---

## Key Differences

| Aspect        | Pipeline Download        | Direct Download        |
|--------------|-------------------------|------------------------|
| Format       | `.jsonl.gz` (compressed) | `.jsonl` (raw)        |
| File Size    | Smaller                 | Much larger           |
| Speed        | Slower (streaming)      | Faster (HTTP)         |
| Row Control  | Yes                     | No                    |
| Integration  | Integrated              | Manual                |

---

## Notes

- For benchmarking:
  - **1M – 50M**: use real data  
  - **100M+**: use synthetic data  

- Dataset size depends heavily on text length (tokens), not just row count.