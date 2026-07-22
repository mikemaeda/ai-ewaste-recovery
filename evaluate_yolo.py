"""Evaluate YOLO detection quality and inference latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model metrics and latency.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/best.pt"))
    parser.add_argument("--data-yaml", type=Path, default=Path("dataset/dataset.yaml"))
    parser.add_argument("--latency-source", type=str, default="0")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/yolo_evaluation.json"))
    return parser.parse_args()


def evaluate_metrics(model: YOLO, data_yaml: Path) -> dict:
    if not data_yaml.exists():
        return {"warning": f"Dataset YAML not found: {data_yaml}"}
    results = model.val(data=str(data_yaml), split="val", verbose=False)
    box = results.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def open_capture(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def measure_latency(model: YOLO, source: str, frames: int) -> dict:
    capture = open_capture(source)
    if not capture.isOpened():
        return {"warning": f"Could not open latency source: {source}"}

    latencies_ms: list[float] = []
    while len(latencies_ms) < frames:
        ok, frame = capture.read()
        if not ok:
            break
        start = time.perf_counter()
        model(frame, verbose=False)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    capture.release()

    if not latencies_ms:
        return {"warning": "No frames captured for latency measurement"}
    return {
        "frames": len(latencies_ms),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "median_latency_ms": statistics.median(latencies_ms),
        "fps": 1000.0 / statistics.mean(latencies_ms),
    }


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    report = {
        "model": str(args.model),
        "dataset": str(args.data_yaml),
        "metrics": evaluate_metrics(model, args.data_yaml),
        "latency": measure_latency(model, args.latency_source, args.frames),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Evaluation saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
