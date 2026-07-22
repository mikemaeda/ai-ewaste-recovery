# Dataset Acquisition And Training Plan

The biggest bottleneck is not code. It is high-quality object-detection labels.
The live dashboard, material database, recovery score, and camera pipeline are
ready, but the final detector needs a real YOLO `best.pt` trained on bounding
boxes.

## Best Public Dataset Leads

Use these as leads, not blind plug-in replacements. Always check license,
labels, and whether the dataset is classification or object detection.

| Source type | Why it helps | Link |
| --- | --- | --- |
| EWasteNet / E-Waste Vision Dataset | E-waste classification baseline and paper context | https://arxiv.org/abs/2311.12823 |
| Roboflow Universe / RF100 | Many public object-detection datasets; useful for PCB/components if license allows | https://arxiv.org/abs/2211.13523 |
| PCBDet / FICS-PCB benchmark | PCB component detection research context | https://arxiv.org/abs/2301.09268 |
| Kaggle e-waste datasets | Good for device-level classification images and extra visual variety | https://www.kaggle.com/datasets |
| Hugging Face datasets | Useful for discovering public image datasets, but many are classification-only | https://huggingface.co/datasets |

## Search Terms To Use

Use these on Kaggle, Hugging Face, and Roboflow:

- `e waste image dataset`
- `electronic waste classification`
- `e-waste object detection`
- `PCB component detection`
- `computer hardware detection`
- `CPU GPU RAM SSD dataset`
- `printed circuit board components YOLO`
- `charger adapter electronics dataset`
- `laptop motherboard dataset`

## Class Priority For Data Collection

Prioritize classes by expected recovery value and research usefulness:

1. `cpu`
2. `connector`
3. `laptop_motherboard`
4. `smartphone`
5. `ram`
6. `gpu`
7. `PCB`
8. `ssd`
9. `charger_power_adapter`
10. `transformer`
11. `cable`
12. `power_supply`
13. `battery`
14. `electric_motor`
15. `hard_drive`

Low-value classes such as `keyboard`, `mouse`, `plastic_housing`, and
`display_screen` are still useful as negative/low-priority examples, but they
should not dominate training.

## What Counts As Usable For YOLO

For detection training, each image needs a matching `.txt` file:

```text
image_001.jpg
image_001.txt
```

Each label line must be:

```text
class_id x_center y_center width height
```

All coordinates must be normalized from 0 to 1.

## Recommended Training Workflow

1. Download or collect images into `data/external/`.
2. Keep a source log with dataset name, URL, license, and retained classes.
3. Move usable images into `data/yolo_raw/`.
4. Label bounding boxes with Label Studio or Roboflow.
5. Export labels in YOLO format.
6. Run:

```powershell
python -m ewaste_research.cli.prepare_yolo_data --source-dir data/yolo_raw
```

7. Train:

```powershell
python -m ewaste_research.cli.train_yolo --data-yaml dataset/dataset.yaml --epochs 80 --device cpu
```

For a GPU machine or Colab, use:

```powershell
python -m ewaste_research.cli.train_yolo --data-yaml dataset/dataset.yaml --epochs 100 --device 0
```

8. Copy the best trained weights to:

```text
artifacts/models/best.pt
```

9. Evaluate:

```powershell
python -m ewaste_research.cli.evaluate_yolo --model artifacts/models/best.pt --data-yaml dataset/dataset.yaml
```

10. Demo:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/windows/run-detector.ps1
```

