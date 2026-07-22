from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

if platform.system() == "Windows" and sys.version_info >= (3, 14):
    raise RuntimeError(
        "TensorFlow does not currently publish Windows wheels for Python 3.14. "
        "Use the Python 3.13 environment described in README.md."
    )

import tensorflow as tf


IMG_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify one new image with the saved MobileNetV2 model."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/best_image_classifier.keras"),
    )
    parser.add_argument("--classes", type=Path, default=Path("artifacts/reports/class_names.json"))
    parser.add_argument("--values", type=Path, default=Path("data/material-values.json"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_image(path: Path) -> tf.Tensor:
    image = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    return tf.expand_dims(array, axis=0)


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)
    if not args.model.exists() or not args.classes.exists():
        raise FileNotFoundError(
            "Train first so artifacts/models/best_image_classifier.keras and "
            "artifacts/reports/class_names.json exist."
        )

    model = tf.keras.models.load_model(args.model)
    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    value_lookup = {}
    if args.values.exists():
        value_lookup = json.loads(args.values.read_text(encoding="utf-8"))

    probabilities = model.predict(load_image(args.image), verbose=0)[0]
    ranked_indices = probabilities.argsort()[::-1][: args.top_k]

    print(f"Image: {args.image}")
    for index in ranked_indices:
        label = class_names[int(index)]
        confidence = float(probabilities[index])
        value_note = value_lookup.get(label, "No recovery note configured.")
        print(f"{label}: {confidence:.1%} confidence")
        print(f"  recovery: {value_note}")


if __name__ == "__main__":
    main()
