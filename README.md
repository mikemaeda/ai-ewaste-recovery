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
- Generates poster/research figures from project artifacts.
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

- `DATASET_SOURCES.md`
- `DATASET_ACQUISITION_PLAN.md`
- `external_dataset_candidates.csv`

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

The recovery taxonomy covers 27 e-waste component classes, including CPUs, connectors, laptop motherboards, smartphones, RAM modules, GPUs, PCBs, cables, tablets, SSDs, hard drives, batteries, power supplies, motors, displays, and plastic housings. See `ewaste_research/taxonomy.py` and `material_values.json`.

## Results

The MobileNetV2 classifier reached 90.7% validation accuracy after 10 epochs. Recovery scores were calculated for 27 e-waste components. The highest-ranked recovery targets were CPUs, connectors, laptop motherboards, smartphones, RAM modules, GPUs, PCBs, cables, tablets, and SSDs.

Generated result files are kept under `artifacts/reports/`, including:

- `evaluation.json`
- `class_names.json`
- `recovery_priority_table.csv`
- dataset-analysis summaries
- poster-ready figures in `artifacts/reports/poster/`

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
python train.py --data-dir data/images
```

Predict one image after training:

```powershell
python predict.py samples/new_item.jpg --model artifacts/models/best_image_classifier.keras
```

Rank material-recovery priorities:

```powershell
python rank_recovery.py
```

Generate poster-ready figures:

```powershell
python make_poster_figures.py
```

Analyze a dataset:

```powershell
python analyze_dataset.py --data-dir data/images
```

Run the experimental live detector:

```powershell
python detect_ewaste.py --source auto --backend auto --width 1280 --height 720 --conf 0.25
```

## YOLO Work In Progress

The YOLO object-detection path is included for continued development. Bounding-box labeling and final detector training are not complete yet. Current detector metrics such as precision, recall, mAP, latency, and FPS should be reported only after a custom detector is trained and evaluated.

See `README_YOLO.md`, `prepare_yolo_data.py`, `train_yolo.py`, `evaluate_yolo.py`, and `train_yolo_colab.ipynb`.

## Repository Structure

```text
.
|-- ewaste_research/              # Shared camera, dashboard, taxonomy, materials, plotting utilities
|-- artifacts/reports/            # Public summary metrics, CSVs, and generated figures
|-- data/                         # Dataset instructions; raw images are ignored
|-- train.py                      # MobileNetV2 classifier training
|-- predict.py                    # Saved-model image prediction
|-- detect_ewaste.py              # Experimental YOLO/camera dashboard
|-- material_values.json          # Material profiles and recovery values
|-- rank_recovery.py              # Recovery-score table generator
|-- make_poster_figures.py        # Poster/research figure generator
`-- README_YOLO.md                # YOLO workflow documentation
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
