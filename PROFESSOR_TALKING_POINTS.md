# Professor Talking Points

Use this when you need to explain the project quickly and confidently.

## 30-Second Summary

This project is an AI-assisted e-waste recovery prototype. A USB camera captures
live video, a YOLO object detector identifies electronic components, and the
detected component is connected to a material database. The system then estimates
material composition, economic value, recovery score, and recommended recovery
priority.

## One-Sentence Pitch

> We are using real-time computer vision to identify e-waste components and
> translate detections into material recovery recommendations.

## What The System Does

- Captures live video from an Arducam / USB camera.
- Runs YOLO object detection.
- Identifies e-waste component classes.
- Looks up material composition from `material_values.json`.
- Calculates a recovery score from value, copper content, and precious metals.
- Displays a real-time dashboard with confidence, value, score, and priority.
- Generates dataset analysis graphs and CSV tables.
- Supports camera troubleshooting and dataset augmentation.

## Current Pipeline

```text
Arducam Camera
-> YOLO Detection
-> Component Class
-> Material Database Lookup
-> Recovery Score
-> Recovery Priority
-> Live Dashboard
```

## Expanded Component Classes

The current taxonomy includes:

- PCB
- cable
- battery
- RAM
- heat sink
- CPU
- GPU
- SSD
- hard drive
- smartphone
- tablet
- display screen
- keyboard
- mouse
- printer
- charger / power adapter
- power supply
- laptop motherboard
- connector / port
- capacitor
- transformer
- fan
- electric motor
- speaker
- optical drive
- mixed metal component
- plastic housing

## Recovery Score

The recovery score combines:

- 50% estimated material value
- 30% copper content
- 20% precious metal content

This turns object detection into a recovery decision tool instead of only a
visual classifier.

## Dataset Work Completed

Original dataset:

```text
23,960 images
```

After augmentation:

```text
27,380 images
```

Low-frequency folders were increased:

```text
heat_sink: 800 images -> 1,500 images
Mobile/smartphone: 839 images -> 1,500 images
Mouse: 800 images -> 1,500 images
PCB: 841 images -> 1,500 images
Television/display_screen: 800 images -> 1,500 images
```

Generated outputs:

```text
artifacts/reports/dataset_analysis_augmented/class_distribution.png
artifacts/reports/dataset_analysis_augmented/class_distribution.csv
artifacts/reports/dataset_analysis_augmented_expanded/class_distribution.png
artifacts/reports/recovery_priority_table.csv
```

Current top recovery targets are CPU, connector/port, laptop motherboard,
smartphone, RAM, GPU, PCB, cable, tablet, and SSD.

## Camera Work Completed

The Arducam / USB camera was verified as:

```text
Device name: USB Camera
Camera index: 0
Backend: default
Resolution: 1280x720
FPS: 30
```

Camera troubleshooting tools were added:

```text
camera_probe.py
arducam_viewer.py
run_arducam_viewer.ps1
run_ewaste_dashboard.ps1
```

## What Is Working

- Camera capture works.
- Camera probe works.
- Dataset analysis works.
- Dataset augmentation works.
- Material database lookup works.
- Recovery score calculation works.
- Dashboard rendering works.
- Fallback YOLO demo mode works.

## What Still Needs Work

The biggest remaining task is training the final custom YOLO detector.

Currently missing:

```text
artifacts/models/best.pt
```

That file will be produced after YOLO training.

The project also still needs more bounding-box labels for:

- CPUs
- GPUs
- SSDs
- HDDs
- RAM
- cables
- power supplies
- laptop motherboards

## Honest Limitation To Say Out Loud

> The software infrastructure is mostly complete, but the final custom YOLO
> detector still depends on finishing the bounding-box dataset and training the
> `best.pt` model. Right now, the system can run in fallback mode, but final
> research metrics require the trained detector.

## Why This Is Still A Strong Research Prototype

This project goes beyond simple image classification. It connects detection to
resource recovery by combining:

- Computer vision
- Dataset analysis
- Data augmentation
- Material composition lookup
- Economic value estimation
- Recovery score calculation
- Real-time camera demonstration

## Commands To Demonstrate

Test camera:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_arducam_viewer.ps1
```

Run dashboard:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_ewaste_dashboard.ps1
```

Analyze dataset:

```powershell
python analyze_dataset.py --data-dir data/images_augmented --output-dir artifacts/reports/dataset_analysis_augmented
```

Evaluate trained YOLO model after `best.pt` exists:

```powershell
python evaluate_yolo.py --model artifacts/models/best.pt --data-yaml dataset/dataset.yaml
```

## Best Way To Explain The Current Stage

> The project is currently at the prototype integration stage. The camera,
> dashboard, material scoring, dataset analysis, and augmentation tools are in
> place. The next research milestone is completing high-quality YOLO annotations
> and training the final detector so we can report precision, recall, mAP, and
> latency.
