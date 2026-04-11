"""
src/data/download_amazon.py
Download Amazon Reviews dataset from public sources.

Amazon Review Data (2023) by Hou et al. is publicly hosted at:
    https://amazon-reviews-2023.github.io/

This script downloads selected category gz files into data/raw/.

Usage:
    python src/data/download_amazon.py --category All_Beauty Electronics Books
    python src/data/download_amazon.py --list-categories
    python src/data/download_amazon.py --all          # download everything (~70 GB)
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import requests
from tqdm import tqdm

from src.core.config import RAW_DIR
from src.utils import get_logger

logger = get_logger("data.download")

# ─────────────────────────────────────────────────────────
# Amazon Review Data 2023 — category listing
# Source: https://amazon-reviews-2023.github.io/
# ─────────────────────────────────────────────────────────
BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"

CATEGORIES = {
    "All_Beauty":                "All_Beauty.jsonl.gz",
    "Amazon_Fashion":            "Amazon_Fashion.jsonl.gz",
    "Appliances":                "Appliances.jsonl.gz",
    "Arts_Crafts_and_Sewing":   "Arts_Crafts_and_Sewing.jsonl.gz",
    "Automotive":                "Automotive.jsonl.gz",
    "Baby_Products":             "Baby_Products.jsonl.gz",
    "Beauty_and_Personal_Care":  "Beauty_and_Personal_Care.jsonl.gz",
    "Books":                     "Books.jsonl.gz",
    "CDs_and_Vinyl":             "CDs_and_Vinyl.jsonl.gz",
    "Cell_Phones_and_Accessories": "Cell_Phones_and_Accessories.jsonl.gz",
    "Clothing_Shoes_and_Jewelry":  "Clothing_Shoes_and_Jewelry.jsonl.gz",
    "Digital_Music":             "Digital_Music.jsonl.gz",
    "Electronics":               "Electronics.jsonl.gz",
    "Gift_Cards":                "Gift_Cards.jsonl.gz",
    "Grocery_and_Gourmet_Food":  "Grocery_and_Gourmet_Food.jsonl.gz",
    "Handmade_Products":         "Handmade_Products.jsonl.gz",
    "Health_and_Household":      "Health_and_Household.jsonl.gz",
    "Home_and_Kitchen":          "Home_and_Kitchen.jsonl.gz",
    "Industrial_and_Scientific": "Industrial_and_Scientific.jsonl.gz",
    "Kindle_Store":              "Kindle_Store.jsonl.gz",
    "Magazine_Subscriptions":    "Magazine_Subscriptions.jsonl.gz",
    "Movies_and_TV":             "Movies_and_TV.jsonl.gz",
    "Musical_Instruments":       "Musical_Instruments.jsonl.gz",
    "Office_Products":           "Office_Products.jsonl.gz",
    "Patio_Lawn_and_Garden":     "Patio_Lawn_and_Garden.jsonl.gz",
    "Pet_Supplies":              "Pet_Supplies.jsonl.gz",
    "Software":                  "Software.jsonl.gz",
    "Sports_and_Outdoors":       "Sports_and_Outdoors.jsonl.gz",
    "Subscription_Boxes":        "Subscription_Boxes.jsonl.gz",
    "Tools_and_Home_Improvement":"Tools_and_Home_Improvement.jsonl.gz",
    "Toys_and_Games":            "Toys_and_Games.jsonl.gz",
    "Video_Games":               "Video_Games.jsonl.gz",
}

# Recommended small categories for quick testing (~< 1 GB each)
SMALL_CATEGORIES = [
    "All_Beauty", "Gift_Cards", "Magazine_Subscriptions",
    "Software", "Subscription_Boxes", "Digital_Music",
]

# ─────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """
    Stream-download a file with a tqdm progress bar.
    Skips if file already exists and size matches Content-Length.
    """
    if dest.exists():
        # Check remote size
        try:
            resp = requests.head(url, timeout=10)
            remote_size = int(resp.headers.get("Content-Length", 0))
            local_size  = dest.stat().st_size
            if remote_size and local_size == remote_size:
                logger.info(f"  Already downloaded: {dest.name} ({local_size/1e6:.1f} MB)")
                return dest
            logger.info(f"  File exists but size mismatch — re-downloading: {dest.name}")
        except Exception:
            logger.info(f"  File exists — skipping (could not verify size): {dest.name}")
            return dest

    logger.info(f"  Downloading: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            ncols=80,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

    logger.info(f"  Saved: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def download_categories(
    categories: list[str],
    dest_dir: Path = RAW_DIR,
    retry: int = 3,
) -> dict[str, Path]:
    """
    Download one or more Amazon category files.

    Args:
        categories : list of category names from CATEGORIES dict
        dest_dir   : local directory to save .gz files
        retry      : number of retry attempts on network error

    Returns:
        dict {category_name: local_path}
    """
    results = {}
    for cat in categories:
        if cat not in CATEGORIES:
            logger.warning(f"Unknown category: '{cat}'. Use --list-categories to see options.")
            continue

        filename = CATEGORIES[cat]
        url  = BASE_URL + filename
        dest = dest_dir / filename

        for attempt in range(1, retry + 1):
            try:
                path = download_file(url, dest)
                results[cat] = path
                break
            except Exception as exc:
                logger.error(f"  Attempt {attempt}/{retry} failed for {cat}: {exc}")
                if attempt < retry:
                    time.sleep(5 * attempt)
                else:
                    logger.error(f"  Giving up on {cat}")

    return results


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews dataset categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a few small categories (fast, ~200 MB total)
  python src/data/download_amazon.py --small

  # Download specific categories
  python src/data/download_amazon.py --category Electronics Books Automotive

  # Download everything (~70 GB, takes hours)
  python src/data/download_amazon.py --all

  # List all available categories
  python src/data/download_amazon.py --list-categories
        """,
    )
    parser.add_argument("--category", nargs="+", help="Specific category names to download")
    parser.add_argument("--small",    action="store_true", help="Download recommended small categories")
    parser.add_argument("--all",      action="store_true", help="Download ALL categories (~70 GB)")
    parser.add_argument("--list-categories", action="store_true", help="Print available categories and exit")
    parser.add_argument("--dest",     default=str(RAW_DIR), help="Destination directory")
    args = parser.parse_args()

    if args.list_categories:
        print("\nAvailable Amazon Review categories:")
        for name, fname in CATEGORIES.items():
            marker = " ← small/recommended" if name in SMALL_CATEGORIES else ""
            print(f"  {name:<40} {fname}{marker}")
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
    downloaded = download_categories(cats, dest_dir=dest)
    logger.info(f"\nDownloaded {len(downloaded)}/{len(cats)} files successfully.")