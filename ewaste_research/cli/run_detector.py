"""
ewaste_research/cli/run_detector.py

Live, on-screen YOLO11 object detection for e-waste, with material-recovery
readouts in the terminal.

WHAT YOU GET
------------
* A window pops up on your screen (OpenCV).
* Source can be your webcam (default) OR a single image / video file.
* Every detected object gets a bounding box + label + confidence drawn on it.
* For each new detection, the terminal prints the object name, confidence, and
  the top-3 recoverable materials with their percentages, plus recovery
  priority and estimated scrap value (read from data/material-values.json).
* Press Q (or Esc) to quit.

TWO WAYS IT RUNS
----------------
1. TRAINED MODE: if your trained weights exist (default artifacts/models/best.pt),
   it detects your 6 e-waste classes directly.
2. FALLBACK MODE: if no trained weights are found, it downloads the pretrained
   YOLO11-nano model (COCO, 80 everyday classes) and maps a handful of those
   classes onto your e-waste categories so you can see the window working
   immediately, before you have trained anything.

EXAMPLES
--------
    # Webcam (camera 0), fallback model if you have not trained yet:
    python -m ewaste_research.cli.run_detector

    # A specific image:
    python -m ewaste_research.cli.run_detector --source path/to/photo.jpg

    # Your trained model on the webcam:
    python -m ewaste_research.cli.run_detector --model artifacts/models/best.pt --source 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from ewaste_research.camera import CAMERA_BACKENDS, open_camera
from ewaste_research.dashboard import render_dashboard
from ewaste_research.materials import calculate_recovery_features, recommended_priority
from ewaste_research.taxonomy import CANONICAL_CLASSES, normalize_component


# Where your trained weights are expected (download best.pt from Colab here).
DEFAULT_MODEL = Path("artifacts/models/best.pt")
# Pretrained model used for fallback mode. Ultralytics downloads it on first use.
FALLBACK_MODEL = "yolo11n.pt"
DEFAULT_VALUES = Path("data/material-values.json")

# Expanded e-waste class names used by the current research taxonomy.
EWASTE_CLASSES = set(CANONICAL_CLASSES)

# Fallback only: map COCO (pretrained) class names onto your e-waste categories.
# This is a rough placeholder so the window does something useful before you
# train your own model. Anything not in this map is shown with its COCO name.
COCO_TO_EWASTE = {
    "cell phone": "smartphone",
    "laptop": "laptop_motherboard",
    "tv": "display_screen",
    "tvmonitor": "display_screen",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "remote": "plastic_housing",
    "microwave": "transformer",
    "oven": "transformer",
    "toaster": "transformer",
    "refrigerator": "electric_motor",
    "clock": "PCB",
}

# A fixed colour (B, G, R) per e-waste class so boxes are easy to tell apart.
CLASS_COLORS = {
    "PCB": (0, 200, 0),
    "cable": (0, 165, 255),
    "battery": (0, 0, 255),
    "ram": (80, 220, 80),
    "heat_sink": (255, 180, 50),
    "cpu": (60, 220, 220),
    "gpu": (120, 180, 255),
    "ssd": (220, 180, 80),
    "hard_drive": (255, 0, 0),
    "smartphone": (70, 210, 255),
    "tablet": (90, 180, 255),
    "display_screen": (170, 150, 255),
    "keyboard": (160, 160, 160),
    "mouse": (130, 130, 130),
    "printer": (180, 180, 80),
    "charger_power_adapter": (50, 190, 230),
    "power_supply": (150, 150, 255),
    "laptop_motherboard": (0, 220, 120),
    "connector": (0, 240, 180),
    "capacitor": (120, 210, 120),
    "transformer": (100, 100, 240),
    "fan": (210, 180, 120),
    "electric_motor": (80, 160, 220),
    "speaker": (180, 130, 210),
    "optical_drive": (220, 220, 120),
    "metal_component": (200, 200, 0),
    "plastic_housing": (200, 0, 200),
}
DEFAULT_COLOR = (180, 180, 180)

# In webcam mode, do not spam the terminal: print a given class at most once
# every PRINT_COOLDOWN_FRAMES frames.
PRINT_COOLDOWN_FRAMES = 45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live YOLO11 e-waste detection with material readouts."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to trained YOLO weights (.pt). Falls back to yolo11n.pt if missing.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        help='Webcam index, "auto", or path to an image/video file.',
    )
    parser.add_argument(
        "--values",
        type=Path,
        default=DEFAULT_VALUES,
        help="Path to data/material-values.json.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum confidence to show a detection (0-1).",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the real-time recovery dashboard side panel.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested camera capture width for webcam/Arducam sources.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested camera capture height for webcam/Arducam sources.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", *sorted(CAMERA_BACKENDS)],
        default="auto",
        help="OpenCV camera backend for webcam/Arducam sources.",
    )
    return parser.parse_args()


def load_material_values(values_path: Path) -> dict:
    """Load data/material-values.json, or return an empty dict if it is missing."""
    if not values_path.exists():
        print(f"WARNING: {values_path} not found. Material readouts disabled.")
        return {}
    return json.loads(values_path.read_text(encoding="utf-8"))


def load_model(model_path: Path) -> tuple[YOLO, bool]:
    """
    Load the trained model if it exists, otherwise the pretrained fallback.

    Returns (model, is_fallback).
    """
    if model_path.exists():
        print(f"Loading trained model: {model_path}")
        return YOLO(str(model_path)), False

    print(
        f"No trained model at {model_path}.\n"
        f"FALLBACK MODE: using pretrained {FALLBACK_MODEL} (COCO classes) and "
        "mapping them onto e-waste categories. Train your own model for real results."
    )
    return YOLO(FALLBACK_MODEL), True


def map_label(raw_label: str, is_fallback: bool) -> str:
    """
    Convert a raw model label into an e-waste class name.

    In trained mode the label already IS an e-waste class. In fallback mode we
    translate COCO names through COCO_TO_EWASTE when possible.
    """
    if not is_fallback:
        return normalize_component(raw_label)
    return normalize_component(COCO_TO_EWASTE.get(raw_label, raw_label))


def print_material_report(
    ewaste_label: str, confidence: float, values: dict
) -> None:
    """Print object name, confidence, and top-3 materials to the terminal."""
    entry = values.get(ewaste_label)

    # Rich entries are dicts with a "materials" list; legacy entries are strings.
    if not isinstance(entry, dict):
        print(f"\n[DETECTED] {ewaste_label}  ({confidence:.0%} confidence)")
        if isinstance(entry, str):
            print(f"  recovery note: {entry}")
        else:
            print("  (no material data configured for this class)")
        return

    display = entry.get("display_name", ewaste_label)
    priority = entry.get("recovery_priority", "unknown")
    value = entry.get("estimated_value_usd_per_kg")
    materials = entry.get("materials", [])
    features = calculate_recovery_features(entry)
    calculated_priority = recommended_priority(features.recovery_score, priority)

    # Sort materials by percentage, highest first, and take the top 3.
    top3 = sorted(materials, key=lambda m: m.get("percent", 0), reverse=True)[:3]

    print(f"\n[DETECTED] {display}  ({confidence:.0%} confidence)")
    print(f"  recovery priority: {calculated_priority}")
    if value is not None:
        print(f"  estimated value:   ${value:.2f} / kg")
    print(f"  recovery score:    {features.recovery_score:.1f}/100")
    print(f"  copper / precious: {features.copper_percent:.2f}% / {features.precious_metal_percent:.3f}%")
    print("  top 3 materials:")
    for material in top3:
        print(f"    - {material.get('material', '?')}: {material.get('percent', 0)}%")


def draw_box(frame, x1, y1, x2, y2, label_text, color) -> None:
    """Draw one bounding box plus a filled label tag onto the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw a filled rectangle behind the text so the label is always readable.
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    top = max(y1, text_h + 6)
    cv2.rectangle(frame, (x1, top - text_h - 6), (x1 + text_w + 4, top), color, -1)
    cv2.putText(
        frame,
        label_text,
        (x1 + 2, top - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def annotate_frame(frame, result, is_fallback, values, conf_thresh, last_printed, frame_idx):
    """
    Draw every detection on the frame and print reports for new detections.

    last_printed maps an e-waste label -> the frame index it was last printed on,
    which throttles terminal spam in webcam mode.
    """
    names = result.names  # id -> raw label name, from the model
    boxes = result.boxes
    if boxes is None:
        return []

    detections: list[dict] = []
    for box in boxes:
        confidence = float(box.conf[0])
        if confidence < conf_thresh:
            continue

        class_id = int(box.cls[0])
        raw_label = names.get(class_id, str(class_id))

        # In fallback mode the model knows 80 everyday COCO objects (person,
        # chair, car, ...). We only care about the ones that map to an e-waste
        # category, so skip everything else. This is what stops it drawing a
        # box around a person.
        if is_fallback and raw_label not in COCO_TO_EWASTE:
            continue

        ewaste_label = map_label(raw_label, is_fallback)

        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        color = CLASS_COLORS.get(ewaste_label, DEFAULT_COLOR)

        # In fallback mode, show what the raw model actually saw, for honesty.
        if is_fallback and raw_label != ewaste_label:
            label_text = f"{ewaste_label} ({raw_label}) {confidence:.0%}"
        else:
            label_text = f"{ewaste_label} {confidence:.0%}"

        draw_box(frame, x1, y1, x2, y2, label_text, color)
        detections.append(
            {
                "label": ewaste_label,
                "raw_label": raw_label,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
            }
        )

        # Print a material report, throttled so the terminal stays readable.
        last = last_printed.get(ewaste_label, -PRINT_COOLDOWN_FRAMES)
        if frame_idx - last >= PRINT_COOLDOWN_FRAMES:
            print_material_report(ewaste_label, confidence, values)
            last_printed[ewaste_label] = frame_idx
    return detections


def is_image_file(source: str) -> bool:
    """True if the source string points to an image file on disk."""
    path = Path(source)
    return path.exists() and path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }


def run_on_image(model, image_path, is_fallback, values, conf_thresh, show_dashboard) -> None:
    """Detect on a single still image and keep it on screen until Q/Esc."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = model(frame, conf=conf_thresh, verbose=False)[0]
    detections = annotate_frame(frame, result, is_fallback, values, conf_thresh, {}, 0)
    if show_dashboard:
        frame = render_dashboard(frame, detections, values)

    window = "e-waste detection (press Q to quit)"
    cv2.imshow(window, frame)
    print("\nImage shown. Press Q or Esc in the window to close.")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        # Also quit if the user clicks the window's X button.
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def run_on_stream(
    model,
    source,
    is_fallback,
    values,
    conf_thresh,
    show_dashboard,
    width,
    height,
    backend,
) -> None:
    """Detect live on a webcam index or a video file until Q/Esc."""
    capture, camera_info = open_camera(source, backend, width, height)

    window = "e-waste detection (press Q to quit)"
    last_printed: dict[str, int] = {}
    frame_idx = 0

    print(
        f"\nOpened camera source {camera_info['source']} with backend "
        f"{camera_info['backend']} at {camera_info.get('width')}x"
        f"{camera_info.get('height')} fps={camera_info.get('fps'):.1f}"
    )
    print("Live window open. Press Q or Esc to quit.")
    while True:
        ok, frame = capture.read()
        if not ok:
            print("End of stream / no frame received.")
            break

        result = model(frame, conf=conf_thresh, verbose=False)[0]
        detections = annotate_frame(
            frame, result, is_fallback, values, conf_thresh, last_printed, frame_idx
        )
        if show_dashboard:
            frame = render_dashboard(frame, detections, values)

        cv2.imshow(window, frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    values = load_material_values(args.values)
    model, is_fallback = load_model(args.model)

    if is_image_file(args.source):
        run_on_image(model, args.source, is_fallback, values, args.conf, not args.no_dashboard)
    else:
        run_on_stream(
            model,
            args.source,
            is_fallback,
            values,
            args.conf,
            not args.no_dashboard,
            args.width,
            args.height,
            args.backend,
        )


if __name__ == "__main__":
    main()
