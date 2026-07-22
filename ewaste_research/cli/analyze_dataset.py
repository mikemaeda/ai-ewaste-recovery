"""Analyze e-waste image or YOLO datasets and generate poster-ready outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from ewaste_research.plots import apply_publication_style
from ewaste_research.taxonomy import CANONICAL_CLASSES, FOLDER_TO_COMPONENT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze class balance for e-waste datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--yolo-dir", type=Path, help="Optional YOLO dataset folder with labels/train and labels/val.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/dataset_analysis"))
    parser.add_argument("--low-frequency-threshold", type=int, default=900)
    return parser.parse_args()


def count_folder_dataset(data_dir: Path) -> Counter:
    counts: Counter[str] = Counter()
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        component = FOLDER_TO_COMPONENT.get(folder.name, folder.name)
        image_count = sum(1 for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        counts[component] += image_count
    return counts


def load_yolo_names(yolo_dir: Path) -> list[str]:
    yaml_path = yolo_dir / "dataset.yaml"
    if not yaml_path.exists():
        return CANONICAL_CLASSES
    names: list[str] = []
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if ":" in stripped and stripped[0].isdigit():
            names.append(stripped.split(":", 1)[1].strip())
    return names or CANONICAL_CLASSES


def count_yolo_labels(yolo_dir: Path) -> Counter:
    names = load_yolo_names(yolo_dir)
    counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        label_dir = yolo_dir / "labels" / split
        if not label_dir.exists():
            continue
        for label_path in label_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                if 0 <= class_id < len(names):
                    counts[names[class_id]] += 1
    return counts


def write_csv(counts: Counter, output_path: Path, threshold: int) -> None:
    total = sum(counts.values())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "count", "percent", "status"])
        for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            percent = (count / total * 100.0) if total else 0.0
            status = "underrepresented" if count < threshold else "ok"
            writer.writerow([name, count, f"{percent:.2f}", status])


def plot_distribution(counts: Counter, output_path: Path, threshold: int) -> None:
    apply_publication_style()
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    colors = ["#c44949" if value < threshold else "#2f6f9f" for value in values]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, values, color=colors)
    ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1, label=f"Low-frequency threshold ({threshold})")
    ax.set_title("E-Waste Dataset Class Distribution")
    ax.set_xlabel("Component class")
    ax.set_ylabel("Images / labels")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.yolo_dir:
        counts = count_yolo_labels(args.yolo_dir)
        source = str(args.yolo_dir)
    else:
        counts = count_folder_dataset(args.data_dir)
        source = str(args.data_dir)

    underrepresented = {
        name: count for name, count in counts.items() if count < args.low_frequency_threshold
    }
    write_csv(counts, args.output_dir / "class_distribution.csv", args.low_frequency_threshold)
    plot_distribution(counts, args.output_dir / "class_distribution.png", args.low_frequency_threshold)
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "source": source,
                "total_examples": sum(counts.values()),
                "class_counts": dict(counts),
                "underrepresented_classes": underrepresented,
                "low_frequency_threshold": args.low_frequency_threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Analyzed {sum(counts.values())} examples from {source}")
    print(f"Underrepresented classes: {underrepresented or 'none'}")
    print(f"Reports saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

