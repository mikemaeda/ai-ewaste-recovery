"""Create augmented images for low-frequency e-waste classes."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balance folder datasets with targeted augmentation.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/images_augmented"))
    parser.add_argument("--target-count", type=int, default=1500)
    parser.add_argument(
        "--only-class",
        action="append",
        default=[],
        help="Folder name to augment. Can be repeated. Defaults to every folder below target-count.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply mild camera-realistic changes for e-waste component images."""
    img = image.convert("RGB")
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    angle = rng.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.18))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.20))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.35:
        img = ImageOps.autocontrast(img)
    return img


def copy_originals(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        dest_dir = output_dir / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_files(class_dir):
            dest = dest_dir / image_path.name
            if not dest.exists():
                shutil.copy2(image_path, dest)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.source_dir.exists():
        raise FileNotFoundError(args.source_dir)

    copy_originals(args.source_dir, args.output_dir)
    counts = Counter({folder.name: len(image_files(folder)) for folder in args.output_dir.iterdir() if folder.is_dir()})

    for class_name, count in sorted(counts.items()):
        if args.only_class and class_name not in args.only_class:
            continue
        class_dir = args.output_dir / class_name
        originals = image_files(class_dir)
        if count >= args.target_count or not originals:
            continue

        needed = args.target_count - count
        print(f"Augmenting {class_name}: {count} -> {args.target_count}")
        for index in range(needed):
            source_image = rng.choice(originals)
            with Image.open(source_image) as image:
                augmented = augment_image(image, rng)
            dest = class_dir / f"{source_image.stem}_aug_{index:04d}.jpg"
            augmented.save(dest, quality=92)

    final_counts = {folder.name: len(image_files(folder)) for folder in args.output_dir.iterdir() if folder.is_dir()}
    print(f"Augmented dataset ready: {args.output_dir.resolve()}")
    for name, count in sorted(final_counts.items(), key=lambda item: item[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
