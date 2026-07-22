"""
End-to-end image classification with TensorFlow, Keras, and MobileNetV2.

Expected dataset structure:

data/images/
    Category_A/
        image_001.jpg
        image_002.jpg
    Category_B/
        image_001.jpg
    Category_C/
        image_001.jpg

Each subfolder name becomes a class label automatically.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# TensorFlow 2.21 has Windows wheels through Python 3.13, but not Python 3.14.
if platform.system() == "Windows" and sys.version_info >= (3, 14):
    raise RuntimeError(
        "This project needs Python 3.13 on Windows. Follow the environment "
        "instructions in README.md, then run this script from that environment."
    )

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Reproducible configuration values used throughout the program.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.20
EPOCHS = 10
RANDOM_SEED = 42


def parse_arguments() -> argparse.Namespace:
    """Read command-line options so paths can be changed without editing code."""
    parser = argparse.ArgumentParser(
        description="Train and test a MobileNetV2 image classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/images"),
        help="Folder containing one subfolder per image category.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Folder in which models, plots, and reports will be saved.",
    )
    parser.add_argument(
        "--predict-image",
        type=Path,
        help="Optional new image to classify after training.",
    )
    return parser.parse_args()


def validate_dataset_directory(data_dir: Path) -> None:
    """Give useful errors before TensorFlow starts building the datasets."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\n"
            "Create folders such as data/images/Category_A and put images inside."
        )

    class_directories = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if len(class_directories) < 2:
        raise ValueError(
            f"{data_dir} must contain at least two category subfolders."
        )


def load_datasets(
    data_dir: Path,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Load images and make a reproducible 80/20 training-validation split.

    label_mode="categorical" creates one-hot encoded labels. For example, with
    three classes, the second class is represented as [0, 1, 0]. This matches
    the categorical cross-entropy loss used when compiling the model.
    """
    common_options = {
        "directory": data_dir,
        "validation_split": VALIDATION_SPLIT,
        "seed": RANDOM_SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "label_mode": "categorical",
    }

    training_dataset = keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common_options,
    )
    validation_dataset = keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common_options,
    )

    class_names = training_dataset.class_names

    # Prefetch overlaps data preparation with model execution for faster training.
    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    return training_dataset, validation_dataset, class_names


def build_model(number_of_classes: int) -> keras.Model:
    """Build a transfer-learning classifier using frozen MobileNetV2 features."""

    # These transformations run only during training. They produce slightly
    # different versions of each image and reduce memorization/overfitting.
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    # MobileNetV2 was previously trained on ImageNet. Removing its original top
    # layer lets us reuse its visual features for our own categories.
    base_model = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    # Freezing prevents the pretrained MobileNetV2 weights from changing.
    # Only the new classification head will learn during these 10 epochs.
    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,), name="input_image")
    x = data_augmentation(inputs)

    # MobileNetV2 expects RGB values scaled from [0, 255] into [-1, 1].
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)

    # Global average pooling converts each feature map into one number and uses
    # far fewer parameters than flattening all spatial feature values.
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    # This custom hidden layer learns combinations of MobileNetV2 features.
    x = layers.Dense(128, activation="relu", name="hidden_dense")(x)
    x = layers.Dropout(0.30, name="dropout")(x)

    # Softmax returns one probability per class, and all probabilities sum to 1.
    outputs = layers.Dense(
        number_of_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ewaste_mobilenet_v2")
    return model


def plot_training_history(
    history: keras.callbacks.History,
    output_path: Path,
) -> None:
    """Plot and save training/validation accuracy and loss."""
    epochs = range(1, len(history.history["accuracy"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history.history["accuracy"], label="Training accuracy")
    axes[0].plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation accuracy",
    )
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.history["loss"], label="Training loss")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Categorical cross-entropy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def predict_single_image(
    model: keras.Model,
    image_path: Path | str,
    class_names: list[str],
) -> tuple[str, float, dict[str, float]]:
    """
    Classify one new image.

    The function returns the predicted label, its confidence, and a dictionary
    containing the probability of every class.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Prediction image not found: {image_path}")

    image = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = keras.utils.img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)

    # Preprocessing is already part of the model, so raw [0, 255] pixels are used.
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    all_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(class_names, probabilities)
    }

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print("All class probabilities:")
    for class_name, probability in sorted(
        all_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {class_name}: {probability:.2%}")

    return predicted_label, confidence, all_probabilities


def main() -> None:
    """Run data loading, model training, evaluation, saving, and prediction."""
    args = parse_arguments()
    validate_dataset_directory(args.data_dir)

    models_directory = args.artifacts_dir / "models"
    reports_directory = args.artifacts_dir / "reports"
    models_directory.mkdir(parents=True, exist_ok=True)
    reports_directory.mkdir(parents=True, exist_ok=True)

    # 1. Load the class-folder dataset and automatically split it 80/20.
    training_dataset, validation_dataset, class_names = load_datasets(args.data_dir)
    print(f"Classes discovered: {class_names}")

    # 2-4. Build the augmented MobileNetV2 model and custom classification head.
    model = build_model(number_of_classes=len(class_names))

    # 5. Adam updates the trainable head weights. Categorical cross-entropy
    # compares the one-hot labels with the model's softmax probabilities.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Preserve the epoch with the lowest validation loss while still running all
    # 10 requested epochs.
    best_model_path = models_directory / "best_image_classifier.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    # 6. Evaluate the best saved epoch on the held-out validation images.
    best_model = keras.models.load_model(best_model_path)
    validation_loss, validation_accuracy = best_model.evaluate(
        validation_dataset,
        verbose=1,
    )

    plot_path = reports_directory / "training_history.png"
    plot_training_history(history, plot_path)

    class_names_path = reports_directory / "class_names.json"
    class_names_path.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    report_path = reports_directory / "evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                "validation_accuracy": float(validation_accuracy),
                "validation_loss": float(validation_loss),
                "epochs": EPOCHS,
                "training_images_percent": 80,
                "validation_images_percent": 20,
                "classes": class_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTraining and evaluation complete.")
    print(f"Validation accuracy: {validation_accuracy:.2%}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Training plot: {plot_path}")
    print(f"Evaluation report: {report_path}")

    # When --predict-image is supplied, test a brand-new image immediately.
    if args.predict_image:
        predict_single_image(best_model, args.predict_image, class_names)


if __name__ == "__main__":
    main()
