from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw class-folder photos into train/val/test folders."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def image_files(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_split(files: list[Path], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, split_dir / source.name)


def main() -> None:
    args = parse_args()
    if not args.raw_dir.exists():
        raise FileNotFoundError(
            "Create folders like data/raw/cables, data/raw/circuit_boards, "
            "and put your photos inside them."
        )
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.test_ratio < 0 or args.test_ratio >= 1:
        raise ValueError("--test-ratio must be between 0 and 1.")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be less than 1.")

    class_dirs = sorted(p for p in args.raw_dir.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Need at least two class folders in data/raw.")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    random.seed(args.seed)
    for class_dir in class_dirs:
        files = image_files(class_dir)
        if len(files) < 5:
            raise ValueError(
                f"{class_dir} only has {len(files)} images. "
                "Aim for at least 20-30 per class for today's prototype."
            )

        random.shuffle(files)
        val_count = max(1, round(len(files) * args.val_ratio))
        test_count = round(len(files) * args.test_ratio)
        val_files = files[:val_count]
        test_files = files[val_count : val_count + test_count]
        train_files = files[val_count + test_count :]

        copy_split(train_files, args.output_dir / "train" / class_dir.name)
        copy_split(val_files, args.output_dir / "val" / class_dir.name)
        if test_files:
            copy_split(test_files, args.output_dir / "test" / class_dir.name)

        print(
            f"{class_dir.name}: {len(train_files)} train, "
            f"{len(val_files)} val, {len(test_files)} test"
        )

    print(f"Prepared dataset in {args.output_dir}")


if __name__ == "__main__":
    main()
