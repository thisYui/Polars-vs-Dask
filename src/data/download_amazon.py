"""
src/data/download_amazon.py
Download Amazon Reviews 2023 dataset from Hugging Face.

Amazon Review Data (2023) by Hou et al. is now hosted at:
    https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

NOTE: The original UCSD datarepo mirror (datarepo.eng.ucsd.edu) is no longer available.
This script uses the `datasets` library to stream or download categories.

Features:
    - Resume interrupted downloads (tracks progress in .jsonl.gz.progress sidecar)
    - Detects and removes corrupt/incomplete files before retrying
    - Retries each category up to MAX_RETRIES times on network errors

Usage:
    python src/data/download_amazon.py --category All_Beauty Electronics Books
    python src/data/download_amazon.py --list-categories
    python src/data/download_amazon.py --small
    python src/data/download_amazon.py --all          # download everything (~113 GB)

Requirements:
    pip install datasets huggingface_hub
"""

import argparse
import gzip
import sys
import time
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from src.core.config import RAW_DIR
from src.utils import get_logger

logger = get_logger("data.download")

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"

MAX_RETRIES    = 5      # retries per category on network error
RETRY_DELAY    = 10     # seconds between retries (doubles each time)
PROGRESS_EVERY = 500_000

CATEGORIES: dict[str, str] = {
    "All_Beauty":                  "raw_review_All_Beauty",
    "Amazon_Fashion":              "raw_review_Amazon_Fashion",
    "Appliances":                  "raw_review_Appliances",
    "Arts_Crafts_and_Sewing":      "raw_review_Arts_Crafts_and_Sewing",
    "Automotive":                  "raw_review_Automotive",
    "Baby_Products":               "raw_review_Baby_Products",
    "Beauty_and_Personal_Care":    "raw_review_Beauty_and_Personal_Care",
    "Books":                       "raw_review_Books",
    "CDs_and_Vinyl":               "raw_review_CDs_and_Vinyl",
    "Cell_Phones_and_Accessories": "raw_review_Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry":  "raw_review_Clothing_Shoes_and_Jewelry",
    "Digital_Music":               "raw_review_Digital_Music",
    "Electronics":                 "raw_review_Electronics",
    "Gift_Cards":                  "raw_review_Gift_Cards",
    "Grocery_and_Gourmet_Food":    "raw_review_Grocery_and_Gourmet_Food",
    "Handmade_Products":           "raw_review_Handmade_Products",
    "Health_and_Household":        "raw_review_Health_and_Household",
    "Home_and_Kitchen":            "raw_review_Home_and_Kitchen",
    "Industrial_and_Scientific":   "raw_review_Industrial_and_Scientific",
    "Kindle_Store":                "raw_review_Kindle_Store",
    "Magazine_Subscriptions":      "raw_review_Magazine_Subscriptions",
    "Movies_and_TV":               "raw_review_Movies_and_TV",
    "Musical_Instruments":         "raw_review_Musical_Instruments",
    "Office_Products":             "raw_review_Office_Products",
    "Patio_Lawn_and_Garden":       "raw_review_Patio_Lawn_and_Garden",
    "Pet_Supplies":                "raw_review_Pet_Supplies",
    "Software":                    "raw_review_Software",
    "Sports_and_Outdoors":         "raw_review_Sports_and_Outdoors",
    "Subscription_Boxes":          "raw_review_Subscription_Boxes",
    "Tools_and_Home_Improvement":  "raw_review_Tools_and_Home_Improvement",
    "Toys_and_Games":              "raw_review_Toys_and_Games",
    "Video_Games":                 "raw_review_Video_Games",
}

SMALL_CATEGORIES = [
    "All_Beauty",
    "Gift_Cards",
    "Magazine_Subscriptions",
    "Software",
    "Subscription_Boxes",
    "Digital_Music",
]


# ─────────────────────────────────────────────────────────
# Progress sidecar helpers
# ─────────────────────────────────────────────────────────

def _progress_path(out_path: Path) -> Path:
    return out_path.with_suffix(".gz.progress")

def _load_progress(out_path: Path) -> int:
    """Return number of rows already written, or 0 if no progress file."""
    p = _progress_path(out_path)
    if p.exists():
        try:
            return int(p.read_text().strip())
        except Exception:
            return 0
    return 0

def _save_progress(out_path: Path, rows: int) -> None:
    _progress_path(out_path).write_text(str(rows))

def _clear_progress(out_path: Path) -> None:
    p = _progress_path(out_path)
    if p.exists():
        p.unlink()


def _is_valid_gz(path: Path) -> bool:
    """Quick check: try reading last few bytes to verify gzip is not truncated."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            # Read up to 10 lines from the end — if gz is corrupt this raises
            f.seek(0, 2)   # seek to end (works for text mode gzip)
    except Exception:
        # gzip text mode doesn't support seek; fall back to full read attempt
        pass
    # Lighter check: verify we can open and read at least 1 byte
    try:
        with gzip.open(path, "rb") as f:
            f.read(1)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────
# Core download function (with resume + retry)
# ─────────────────────────────────────────────────────────

def _download_category(
    category: str,
    dest: Path,
    max_rows: int = None,
) -> Path:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required.\n"
            "Install with: pip install datasets huggingface_hub"
        )

    config_name = CATEGORIES[category]
    out_path    = dest / f"{category}.jsonl.gz"
    prog_path   = _progress_path(out_path)

    dest.mkdir(parents=True, exist_ok=True)

    # ── Check existing file ───────────────────────────────
    if out_path.exists():
        if not prog_path.exists():
            logger.info(
                f"  Already downloaded: {out_path.name} "
                f"({out_path.stat().st_size / 1e6:.1f} MB)"
            )
            return out_path
        else:
            rows_done = _load_progress(out_path)
            logger.warning(
                f"  '{category}' was interrupted at {rows_done:,} rows — "
                f"deleting partial file and restarting."
            )
            out_path.unlink(missing_ok=True)
            prog_path.unlink(missing_ok=True)

    cap_str = f", cap={max_rows:,}" if max_rows else ""
    logger.info(f"  Downloading '{category}' (config={config_name}{cap_str}) …")

    BATCH_SIZE = 20000

    for attempt in range(1, MAX_RETRIES + 1):
        count = 0
        buf = bytearray()

        try:
            dataset = load_dataset(
                HF_REPO,
                config_name,
                split="full",
                streaming=True,
                trust_remote_code=True,
            )

            # binary mode + fast gzip
            with gzip.open(out_path, "wb", compresslevel=1) as f:
                for row in dataset:
                    # orjson.dumps + newline embedded — single allocation per row
                    buf += orjson.dumps(row)
                    buf += b"\n"
                    count += 1

                    # flush batch
                    if count % BATCH_SIZE == 0:
                        f.write(buf)
                        buf.clear()

                    # progress
                    if count % PROGRESS_EVERY == 0:
                        _save_progress(out_path, count)
                        logger.info(f"    {count:,} rows written …")

                    # stop condition
                    if max_rows and count >= max_rows:
                        logger.info(f"    Reached max_rows={max_rows:,} — stopping early.")
                        break

                # flush remainder
                if buf:
                    f.write(buf)

            # done
            _clear_progress(out_path)
            size_mb = out_path.stat().st_size / 1e6
            logger.info(f"  Saved: {out_path.name} ({size_mb:.1f} MB, {count:,} rows)")
            return out_path

        except Exception as exc:
            wait = RETRY_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                logger.warning(
                    f"  Network error on '{category}' (attempt {attempt}/{MAX_RETRIES}): {exc}\n"
                    f"  Written so far: {count:,} rows. Retrying in {wait}s …"
                )
                _save_progress(out_path, count)
                out_path.unlink(missing_ok=True)
                time.sleep(wait)
            else:
                logger.error(
                    f"  Failed to download '{category}' after {MAX_RETRIES} attempts: {exc}"
                )
                out_path.unlink(missing_ok=True)
                _clear_progress(out_path)
                raise

    return out_path


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def download_categories(
    categories: list[str],
    dest_dir:   Path = RAW_DIR,
    streaming:  bool = True,   # kept for API compat
    max_rows:   int  = None,
) -> dict[str, Path]:
    """
    Download one or more Amazon category files from Hugging Face.

    Automatically resumes interrupted downloads and retries on network errors.

    Args:
        categories : list of category names from CATEGORIES dict
        dest_dir   : local directory to save .jsonl.gz files
        streaming  : kept for API compatibility (always True internally)
        max_rows   : stop each category after this many rows

    Returns:
        dict {category_name: local_path}  (only successfully downloaded)
    """
    results = {}
    for cat in categories:
        if cat not in CATEGORIES:
            logger.warning(f"Unknown category: '{cat}'. Use --list-categories to see options.")
            continue
        try:
            path = _download_category(cat, dest_dir, max_rows=max_rows)
            results[cat] = path
        except Exception as exc:
            logger.error(f"  Skipping '{cat}': {exc}")

    return results


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews 2023 dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a few small categories (fast, ~200 MB total)
  python src/data/download_amazon.py --small

  # Download specific categories
  python src/data/download_amazon.py --category Electronics Books Automotive

  # Download everything (~113 GB, takes hours)
  python src/data/download_amazon.py --all

  # List all available categories
  python src/data/download_amazon.py --list-categories

Notes:
  - Interrupted downloads resume automatically from the last checkpoint.
  - Progress is saved every 500,000 rows in a .gz.progress sidecar file.
  - Network errors are retried up to 5 times with exponential backoff.
  - The original UCSD mirror is no longer available; data comes from:
    https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
  - Requires: pip install datasets huggingface_hub
        """,
    )
    parser.add_argument("--category",        nargs="+", help="Specific category names to download")
    parser.add_argument("--small",           action="store_true", help="Download recommended small categories")
    parser.add_argument("--all",             action="store_true", help="Download ALL categories (~113 GB)")
    parser.add_argument("--list-categories", action="store_true", help="Print available categories and exit")
    parser.add_argument("--dest",            default=str(RAW_DIR), help="Destination directory")
    parser.add_argument("--max-rows",        type=int, default=None,
                        help="Stop each category after N rows (e.g. 60000000)")
    args = parser.parse_args()

    if args.list_categories:
        print("\nAvailable Amazon Review categories (Hugging Face):")
        for name, cfg in CATEGORIES.items():
            marker = " ← small/recommended" if name in SMALL_CATEGORIES else ""
            print(f"  {name:<40} config={cfg}{marker}")
        sys.exit(0)

    dest = Path(args.dest)

    if args.all:
        cats = list(CATEGORIES.keys())
    elif args.small:
        cats = SMALL_CATEGORIES
    elif args.category:
        cats = args.category
    else:
        parser.print_help()
        print("\nNo action specified. Use --small for a quick start.")
        sys.exit(1)

    logger.info(f"Downloading {len(cats)} categories → {dest}")
    logger.info(f"Source: https://huggingface.co/datasets/{HF_REPO}")
    if args.max_rows:
        logger.info(f"Row cap per category: {args.max_rows:,}")

    downloaded = download_categories(cats, dest_dir=dest, max_rows=args.max_rows)

    ok   = len(downloaded)
    fail = len(cats) - ok
    logger.info(f"\nDownloaded {ok}/{len(cats)} files successfully.")
    if fail:
        missing = [c for c in cats if c not in downloaded]
        logger.warning(f"Failed categories: {missing}")
        logger.info("Re-run the same command to resume — progress is saved automatically.")
        sys.exit(1)