import pyarrow.parquet as pq
from pathlib import Path

# Đường dẫn folder
folder_path = Path(r"Y:\Python\Polar vs Dask\data\benchmark_syn\1M")

total_mem_bytes = 0
file_count = 0

# Duyệt qua tất cả các file .parquet trong folder
for file in folder_path.glob("*.parquet"):
    # Chỉ đọc metadata để lấy schema/cấu trúc hoặc đọc table tùy nhu cầu
    # Ở đây chúng ta dùng read_table để lấy nbytes chính xác nhất như bạn muốn
    table = pq.read_table(file)

    total_mem_bytes += table.nbytes
    file_count += 1
    print(f"Đã xử lý: {file.name} | RAM: {table.nbytes / 1024 ** 2:.2f} MB")

# Quy đổi đơn vị
total_gb = total_mem_bytes / 1024 ** 3

print("-" * 30)
print(f"Tổng số file: {file_count}")
print(f"Tổng kích thước dự kiến trong RAM: {total_gb:.4f} GB")
# import polars as pl
# import gzip
#
# path = r"Y:\Python\Polar vs Dask\data\raw\Clothing_Shoes_and_Jewelry.jsonl.gz"
#
# # Đọc 5 dòng đầu tiên từ file nén
# lines = []
# with gzip.open(path, "rb") as f:
#     for _ in range(5):
#         line = f.readline()
#         if not line:
#             break
#         lines.append(line)
#
# # Chuyển thành DataFrame
# df_head = pl.read_ndjson(b"".join(lines))
#
# # Cấu hình để in ra toàn bộ số cột và độ rộng cột (không bị dấu ...)
# with pl.Config(tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=50):
#     print("--- 5 Dòng đầu tiên của dữ liệu ---")
#     print(df_head)
#
# print("\n--- Cấu trúc chi tiết (Schema) ---")
# print(df_head.schema)


# import pyarrow.parquet as pq
# from pathlib import Path
#
# folder_path = Path(r"Y:\Python\Polar vs Dask\data\benchmark_real\50M")
# total_rows = 0
#
# for file in folder_path.glob("*.parquet"):
#     metadata = pq.read_metadata(file)
#     total_rows += metadata.num_rows
#     print(f"{file.name}: {metadata.num_rows:,} dòng")
#
# print(f"--- Tổng cộng: {total_rows:,} dòng ---")


