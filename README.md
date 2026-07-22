# AI-Assisted E-Waste Recovery

This undergraduate research project connects computer vision with electronic-waste material recovery. The system classifies or detects e-waste components, links the predicted component to a material-profile database, calculates an interpretable recovery score, and ranks components by recycling priority.

The project is intentionally split into completed, experimental, and planned work. The MobileNetV2 image classifier and recovery-scoring layer are implemented. The YOLO object-detection and real-time sorting workflow is included as experimental work in progress.

## Research Motivation

Electronic waste contains recoverable materials such as copper, aluminum, gold, silver, palladium, lithium, cobalt, nickel, and rare-earth materials. Sorting remains a practical bottleneck: a model that only answers "what object is this?" is less useful than a system that also answers "which component should be recovered first?"

## Research Question

Can a computer-vision system identify e-waste components and prioritize them for material recovery using an interpretable recovery scoring model?

## Current Capabilities

- Trains a MobileNetV2 transfer-learning classifier on folder-organized e-waste images.
- Saves model artifacts, class labels, evaluation metrics, and training plots.
- Predicts a new image with the saved classifier.
- Maintains a material-profile database with estimated economic value, copper content, precious-metal content, and recovery priority.
- Calculates a normalized recovery score from material-value features.
- Generates reusable dataset-analysis and recovery-ranking reports.
- Provides an experimental YOLO11 live-detection dashboard for camera or image input.
- Includes dataset analysis and targeted augmentation utilities.

## Technical Workflow

```text
Camera / Image
-> Computer Vision Model
-> Component Class
-> Material Database Lookup
-> Recovery Score
-> Priority Recommendation
```

The recovery score uses:

```text
score = 0.50 * economic value + 0.30 * copper content + 0.20 * precious-metal content
```

Each term is normalized to a 0-100 scale before weighting.

## Model Architecture

The completed classifier uses MobileNetV2 with ImageNet pretrained weights as a frozen feature extractor. A custom classification head applies global average pooling, a 128-neuron dense layer with ReLU activation, dropout, and a final softmax layer.

The experimental detector uses YOLO11 through Ultralytics. If a trained `artifacts/models/best.pt` file is missing, the live detector can run in fallback mode with generic YOLO11 weights, but that mode is only a demonstration and should not be treated as final e-waste detection performance.

## Dataset

The classifier was trained using the Kaggle `E-Waste Image Classification Dataset (18 Classes)` by Harshad S. Gore. The local training dataset is not committed because it is large. Dataset download and planning notes are in:

- `docs/dataset-sources.md`
- `docs/dataset-acquisition.md`
- `data/external-dataset-candidates.csv`

Expected image-classification folder layout:

```text
data/images/
  Battery/
  Keyboard/
  PCB/
  ...
```

The working dataset artifacts report:

- Original dataset: 23,960 images
- Expanded augmented dataset: 27,380 images
- Canonical classes after expanded augmentation: 13
- No canonical class below the 900-image low-frequency threshold after expanded augmentation

## Supported Component Taxonomy

The recovery taxonomy covers 27 e-waste component classes, including CPUs, connectors, laptop motherboards, smartphones, RAM modules, GPUs, PCBs, cables, tablets, SSDs, hard drives, batteries, power supplies, motors, displays, and plastic housings. See `ewaste_research/taxonomy.py` and `data/material-values.json`.

## Results

The MobileNetV2 classifier reached 90.7% validation accuracy after 10 epochs. Recovery scores were calculated for 27 e-waste components. The highest-ranked recovery targets were CPUs, connectors, laptop motherboards, smartphones, RAM modules, GPUs, PCBs, cables, tablets, and SSDs.

Generated result files are kept under `artifacts/reports/`, including:

- `evaluation.json`
- `class_names.json`
- `recovery_priority_table.csv`
- dataset-analysis summaries

## Installation

Create and activate a Python environment. On Windows, TensorFlow currently requires a compatible Python version such as Python 3.13 for this project.

```powershell
python -m pip install -r requirements.txt
```

For YOLO/camera experiments:

```powershell
python -m pip install -r requirements_yolo.txt
```

## Usage

Train the MobileNetV2 classifier:

```powershell
python -m ewaste_research.cli.train_classifier --data-dir data/images
```

Predict one image after training:

```powershell
python -m ewaste_research.cli.predict_classifier samples/new_item.jpg --model artifacts/models/best_image_classifier.keras
```

Rank material-recovery priorities:

```powershell
python -m ewaste_research.cli.rank_recovery
```

Analyze a dataset:

```powershell
python -m ewaste_research.cli.analyze_dataset --data-dir data/images
```

Run the experimental live detector:

```powershell
python -m ewaste_research.cli.run_detector --source auto --backend auto --width 1280 --height 720 --conf 0.25
```

## YOLO Work In Progress

The YOLO object-detection path is included for continued development. Bounding-box labeling and final detector training are not complete yet. Current detector metrics such as precision, recall, mAP, latency, and FPS should be reported only after a custom detector is trained and evaluated.

See `docs/yolo-workflow.md`, `ewaste_research/cli/prepare_yolo_data.py`, `ewaste_research/cli/train_yolo.py`, `ewaste_research/cli/evaluate_yolo.py`, and `notebooks/yolo-training-colab.ipynb`.

## Repository Structure

```text
.
|-- ewaste_research/              # Reusable package code
|   `-- cli/                      # Runnable commands for training, analysis, and detection
|-- data/                         # Reference data and local-dataset instructions
|-- docs/                         # Dataset and YOLO workflow documentation
|-- notebooks/                    # Colab and experiment notebooks
|-- scripts/windows/              # PowerShell setup and launch helpers
|-- artifacts/reports/            # Public metrics, tables, and plots
|-- requirements.txt              # Classifier dependencies
`-- requirements_yolo.txt         # Detector and camera dependencies
```

## Known Limitations

- Material values are curated estimates from published/reference sources and should be validated against measured samples.
- The classifier taxonomy and full 27-component recovery taxonomy are not fully unified.
- The custom YOLO detector is not fully trained yet.
- Raw datasets and model weights are intentionally excluded from GitHub.

## Future Work

- Label bounding boxes for the expanded component taxonomy.
- Train and evaluate the custom YOLO detector.
- Report precision, recall, mAP, latency, and FPS.
- Unify classifier classes with the full material-recovery taxonomy.
- Validate material composition estimates against measured samples.
- Extend the real-time dashboard toward sorting-line deployment.

## License

No license file has been selected yet. Until a license is added, all rights are reserved by default.

## Acknowledgements

This project uses TensorFlow/Keras, MobileNetV2, Ultralytics YOLO, OpenCV, NumPy, Pillow, Matplotlib, and the Kaggle e-waste image-classification dataset referenced above.
