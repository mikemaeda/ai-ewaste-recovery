# Complete MobileNetV2 Image Classifier

This project trains an image classifier from a local folder using TensorFlow, Keras, data augmentation, and MobileNetV2 transfer learning.

## 1. Python Environment

This project already contains a local Python 3.13 runtime in `.python313`. TensorFlow 2.21, Keras, Matplotlib, NumPy, and Pillow are installed in it.

In VS Code, select this interpreter:

```text
C:\Users\mhm5\Desktop\train\.python313\python.exe
```

All commands below use that interpreter explicitly. If dependencies ever need to be reinstalled:

```powershell
.\.python313\python.exe -m pip install -r requirements.txt
```

## 2. Organize The Images

Create one subfolder for each category. The folder names become the model's labels automatically:

```text
data/
  images/
    cables/
      cable_01.jpg
      cable_02.jpg
    circuit_boards/
      board_01.jpg
      board_02.jpg
    smartphones/
      phone_01.jpg
      phone_02.jpg
```

Use at least two categories. For a first prototype, aim for 20-30 varied images per category. Do not put the same or nearly identical photograph in multiple categories.

No manual split is required. `train.py` automatically uses 80% of every class for training and 20% for validation.

### Dataset Currently Installed

The Kaggle `E-Waste Image Classification Dataset (18 Classes)` has already
been downloaded into:

```text
data/kaggle_download/data/
```

It includes the original `train`, `val`, and `test` folders. `data/images` is
a junction to the downloaded `train` folder, so no second 1.6 GB copy is
needed. The training script sees 23,960 images and automatically creates:

```text
19,168 training images (80%)
4,792 validation images (20%)
```

The detected classes are:

```text
Air-Conditioner
Battery
Keyboard
Laptop
Microchip-IC
Microwave
Mobile
Mouse
PCB
Passive-Component
Printer
Refrigerator
Resistor
Television
Washing Machine
heat-sink
light bulbs
transistor
```

## 3. Train And Evaluate

```powershell
.\.python313\python.exe train.py --data-dir data/images
```

The script:

1. Loads and resizes images to 224 x 224 RGB pixels.
2. Creates a reproducible 80/20 training-validation split.
3. Augments training images with flips, rotations, and zoom.
4. loads ImageNet-trained MobileNetV2 without its original classifier.
5. Freezes MobileNetV2 and trains a new ReLU/Softmax classification head.
6. Uses Adam and categorical cross-entropy for 10 epochs.
7. Evaluates the best epoch and saves the model, metrics, and graphs.

Generated files:

```text
artifacts/models/best_image_classifier.keras
artifacts/reports/class_names.json
artifacts/reports/evaluation.json
artifacts/reports/training_history.png
```

## 4. Train And Test A New Image

Supply a photograph that is not part of the training folder:

```powershell
.\.python313\python.exe train.py --data-dir data/images --predict-image samples/new_item.jpg
```

The script prints the predicted category, confidence, and probability for every category.

## 5. Test Later Without Retraining

After the model has been trained, use:

```powershell
.\.python313\python.exe predict.py samples/new_item.jpg `
  --model artifacts/models/best_image_classifier.keras
```

## Architecture Explanation

MobileNetV2 is the feature extractor. Its pretrained convolutional layers recognize useful visual patterns such as edges, textures, and shapes. Those layers are frozen, so their weights do not change.

`GlobalAveragePooling2D` compresses MobileNetV2's spatial feature maps into a feature vector. A 128-neuron Dense layer with ReLU learns combinations relevant to this dataset. Dropout reduces overfitting. The final Dense layer has one Softmax neuron per category and produces probabilities that sum to 1.

Categorical cross-entropy measures the difference between the correct one-hot class label and the Softmax probabilities. Adam updates only the custom classification head during training.
