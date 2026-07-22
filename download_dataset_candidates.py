"""Print Kaggle/Roboflow search terms and optional Kaggle CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

SEARCH_TERMS = [
    "e waste image dataset",
    "electronic waste image classification",
    "printed circuit board object detection",
    "pcb components detection",
    "computer parts detection",
    "hard drive SSD GPU CPU dataset",
    "recycling waste computer components",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan e-waste dataset downloads.")
    parser.add_argument(
        "--kaggle-slug",
        action="append",
        default=[],
        help="Kaggle dataset slug such as owner/dataset-name. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/external"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Search these terms on Kaggle and Roboflow:")
    for term in SEARCH_TERMS:
        print(f"  - {term}")

    if args.kaggle_slug:
        print("\nKaggle CLI download commands:")
        for slug in args.kaggle_slug:
            target = args.output_dir / slug.replace("/", "__")
            print(f"  kaggle datasets download -d {slug} -p {target} --unzip")
    else:
        print("\nAfter you find Kaggle dataset slugs, run for example:")
        print("  python download_dataset_candidates.py --kaggle-slug owner/dataset-name")


if __name__ == "__main__":
    main()

