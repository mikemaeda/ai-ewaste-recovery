"""Find connected cameras and save a test frame for the Arducam/demo setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

BACKENDS = {
    "default": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe camera indexes and save test frames.")
    parser.add_argument("--max-index", type=int, default=5, help="Probe camera indexes 0..N.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/camera_probe"))
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        help="Probe only one OpenCV backend. Defaults to all common Windows backends.",
    )
    return parser.parse_args()


def probe_camera(index: int, backend_name: str, width: int, height: int, output_dir: Path) -> dict:
    capture = cv2.VideoCapture(index, BACKENDS[backend_name])
    if not capture.isOpened():
        return {"index": index, "backend": backend_name, "available": False}

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok = False
    frame = None
    for _ in range(10):
        ok, frame = capture.read()
        if ok and frame is not None:
            break

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()

    result = {
        "index": index,
        "backend": backend_name,
        "available": bool(ok),
        "width": actual_width,
        "height": actual_height,
        "fps": fps,
    }
    if ok and frame is not None:
        image_path = output_dir / f"camera_{index}_{backend_name}_snapshot.jpg"
        cv2.imwrite(str(image_path), frame)
        result["snapshot"] = str(image_path)
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend_names = [args.backend] if args.backend else ["default", "msmf", "dshow"]
    results = []
    for backend_name in backend_names:
        for index in range(args.max_index + 1):
            results.append(
                probe_camera(index, backend_name, args.width, args.height, args.output_dir)
            )
    available = [result for result in results if result["available"]]

    report_path = args.output_dir / "camera_probe.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    if available:
        best = available[0]
        print(f"\nUse this source first: --source {best['index']} --backend {best['backend']}")
        print(f"Snapshot saved: {best.get('snapshot')}")
    else:
        print("\nNo working camera indexes found. Close other camera apps and retry.")
    print(f"Probe report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
