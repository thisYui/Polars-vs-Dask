"""
src/data/compress_jsonl.py
Nén tất cả .jsonl trong data/raw/ thành .jsonl.gz rồi xóa file gốc.

- Bỏ qua file đã có .jsonl.gz tương ứng
- Xóa .jsonl gốc sau khi nén thành công và verify
- Hỗ trợ multi-thread để nén song song nhiều file

Usage:
    python src/data/compress_jsonl.py
    python src/data/compress_jsonl.py --input data/raw
    python src/data/compress_jsonl.py --input data/raw --workers 4
    python src/data/compress_jsonl.py --keep-original   # nén nhưng không xóa gốc
    python src/data/compress_jsonl.py --dry-run         # chỉ xem, không làm gì
"""

import argparse
import gzip
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

try:
    from src.utils import get_logger
    logger = get_logger("data.compress")
except Exception:
    import logging
    logger = logging.getLogger("data.compress")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


# ─────────────────────────────────────────────────────────
# Core compress logic
# ─────────────────────────────────────────────────────────

def _verify_gz(gz_path: Path) -> bool:
    """Verify gzip file có thể đọc được (không bị corrupt)."""
    try:
        with gzip.open(gz_path, "rb") as f:
            # Đọc từng chunk 1MB để verify toàn bộ file
            while f.read(1024 * 1024):
                pass
        return True
    except Exception:
        return False


def compress_file(
    jsonl_path: Path,
    keep_original: bool = False,
    dry_run: bool = False,
) -> tuple[str, float, float, bool]:
    """
    Nén một file .jsonl thành .jsonl.gz rồi xóa file gốc.

    Returns:
        (filename, src_mb, dst_mb, success)
    """
    gz_path = jsonl_path.with_suffix(".jsonl.gz")
    # Nếu tên file là .jsonl thì suffix chỉ là .jsonl → cần xử lý đúng
    # ví dụ: Electronics.jsonl → Electronics.jsonl.gz
    gz_path = jsonl_path.parent / (jsonl_path.name + ".gz")

    src_mb = jsonl_path.stat().st_size / 1024 ** 2

    # Đã có gz rồi → skip
    if gz_path.exists():
        logger.info(f"  [skip] {jsonl_path.name} → {gz_path.name} đã tồn tại")
        if not keep_original and not dry_run:
            jsonl_path.unlink()
            logger.info(f"  [clean] Đã xóa file gốc: {jsonl_path.name}")
        return jsonl_path.name, src_mb, gz_path.stat().st_size / 1024 ** 2, True

    if dry_run:
        logger.info(f"  [dry-run] Sẽ nén: {jsonl_path.name} ({src_mb:.1f} MB) → {gz_path.name}")
        return jsonl_path.name, src_mb, 0.0, True

    logger.info(f"  Đang nén: {jsonl_path.name} ({src_mb:.1f} MB) ...")
    t0 = time.perf_counter()

    try:
        # compresslevel=1: nhanh nhất, tỷ lệ nén ~60-70% với JSONL
        with open(jsonl_path, "rb") as f_in:
            with gzip.open(gz_path, "wb", compresslevel=1) as f_out:
                shutil.copyfileobj(f_in, f_out, length=8 * 1024 * 1024)  # 8MB buffer

        dst_mb  = gz_path.stat().st_size / 1024 ** 2
        elapsed = time.perf_counter() - t0
        ratio   = (1 - dst_mb / src_mb) * 100 if src_mb > 0 else 0

        logger.info(
            f"  ✓ {gz_path.name} | {src_mb:.1f} MB → {dst_mb:.1f} MB "
            f"(-{ratio:.0f}%) | {elapsed:.1f}s"
        )

        # Verify trước khi xóa
        logger.info(f"  Verifying {gz_path.name} ...")
        if not _verify_gz(gz_path):
            logger.error(f"  ✗ Verify FAILED cho {gz_path.name} — giữ nguyên file gốc!")
            gz_path.unlink(missing_ok=True)
            return jsonl_path.name, src_mb, 0.0, False

        logger.info(f"  ✓ Verify OK")

        # Xóa file gốc nếu không có --keep-original
        if not keep_original:
            jsonl_path.unlink()
            logger.info(f"  [clean] Đã xóa file gốc: {jsonl_path.name}")

        return jsonl_path.name, src_mb, dst_mb, True

    except Exception as exc:
        logger.error(f"  ✗ Lỗi khi nén {jsonl_path.name}: {exc}")
        gz_path.unlink(missing_ok=True)  # dọn file gz dở
        return jsonl_path.name, src_mb, 0.0, False


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def compress_all(
    input_dir:     Path,
    keep_original: bool = False,
    workers:       int  = 2,
    dry_run:       bool = False,
) -> dict[str, bool]:
    """
    Nén tất cả .jsonl trong input_dir thành .jsonl.gz.

    Args:
        input_dir     : thư mục chứa file .jsonl (data/raw/)
        keep_original : nếu True, giữ lại file .jsonl gốc sau khi nén
        workers       : số luồng song song
        dry_run       : chỉ liệt kê, không thực thi

    Returns:
        {filename: success_bool}
    """
    files = sorted(input_dir.glob("*.jsonl"))

    if not files:
        logger.info(f"Không tìm thấy file .jsonl nào trong {input_dir}")
        return {}

    # Phân loại
    already_gz  = [f for f in files if (f.parent / (f.name + ".gz")).exists()]
    need_compress = [f for f in files if f not in already_gz]

    logger.info(f"\n{'='*55}")
    logger.info(f"  COMPRESS JSONL → GZ")
    logger.info(f"{'='*55}")
    logger.info(f"  Thư mục   : {input_dir}")
    logger.info(f"  Tổng .jsonl: {len(files)}")
    logger.info(f"  Đã có .gz : {len(already_gz)} (sẽ xóa .jsonl gốc nếu còn)")
    logger.info(f"  Cần nén   : {len(need_compress)}")
    logger.info(f"  Workers   : {workers}")
    logger.info(f"  Giữ gốc   : {keep_original}")
    logger.info(f"  Dry run   : {dry_run}")
    logger.info(f"{'='*55}\n")

    if not files:
        return {}

    results: dict[str, bool] = {}
    total_src = total_dst = 0.0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(compress_file, f, keep_original, dry_run): f
            for f in files
        }
        for fut in as_completed(futures):
            name, src_mb, dst_mb, ok = fut.result()
            results[name] = ok
            total_src += src_mb
            total_dst += dst_mb

    # Summary
    ok_count   = sum(1 for v in results.values() if v)
    fail_count = len(results) - ok_count

    logger.info(f"\n{'='*55}")
    logger.info(f"  KẾT QUẢ")
    logger.info(f"{'='*55}")
    logger.info(f"  Thành công : {ok_count}/{len(results)}")
    if fail_count:
        logger.warning(f"  Thất bại   : {fail_count}")
    if total_src > 0 and not dry_run:
        saved = total_src - total_dst
        logger.info(
            f"  Dung lượng : {total_src:.1f} MB → {total_dst:.1f} MB "
            f"(tiết kiệm {saved:.1f} MB, -{saved/total_src*100:.0f}%)"
        )
    logger.info(f"{'='*55}\n")

    return results


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nén .jsonl → .jsonl.gz và xóa file gốc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Nén tất cả .jsonl trong data/raw/, xóa gốc (mặc định)
  python src/data/compress_jsonl.py

  # Nén nhưng giữ lại .jsonl gốc
  python src/data/compress_jsonl.py --keep-original

  # Chỉ xem sẽ làm gì, không thực thi
  python src/data/compress_jsonl.py --dry-run

  # Nén song song 4 luồng
  python src/data/compress_jsonl.py --workers 4
        """,
    )
    parser.add_argument("--input",         default="data/raw",  help="Thư mục chứa .jsonl")
    parser.add_argument("--keep-original", action="store_true", help="Giữ .jsonl gốc sau khi nén")
    parser.add_argument("--workers",       type=int, default=2, help="Số luồng song song")
    parser.add_argument("--dry-run",       action="store_true", help="Chỉ xem, không nén")
    args = parser.parse_args()

    results = compress_all(
        input_dir=Path(args.input),
        keep_original=args.keep_original,
        workers=args.workers,
        dry_run=args.dry_run,
    )

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.error(f"Các file thất bại: {failed}")
        sys.exit(1)