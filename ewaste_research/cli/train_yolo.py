"""Train the custom e-waste YOLO detector from a prepared YOLO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on the e-waste dataset.")
    parser.add_argument("--data-yaml", type=Path, default=Path("dataset/dataset.yaml"))
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cpu", help='Use "0" for first GPU, or "cpu".')
    parser.add_argument("--project", default="artifacts/yolo_runs")
    parser.add_argument("--name", default="ewaste_detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data_yaml}. "
            "Run `python -m ewaste_research.cli.prepare_yolo_data` after creating YOLO labels."
        )

    model = YOLO(args.base_model)
    results = model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        plots=True,
        cos_lr=True,
        close_mosaic=10,
        degrees=10.0,
        translate=0.10,
        scale=0.50,
        fliplr=0.50,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.35,
    )
    print(results)
    print(
        "\nAfter training, copy the best weights from the run folder to "
        "artifacts/models/best.pt for the live dashboard."
    )


if __name__ == "__main__":
    main()
