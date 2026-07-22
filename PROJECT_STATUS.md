# AI-Assisted E-Waste Recovery Project Status

This document explains what has been built, what is working now, what still
needs to be completed, and how to explain the project clearly to professors.

## 1. Project Goal

The goal of this undergraduate research prototype is to use computer vision to
identify electronic waste components and connect each detected component to
material recovery information.

The intended end-to-end pipeline is:

```text
USB / Arducam Camera
-> YOLO Object Detection
-> Component Classification
-> Material Database Lookup
-> Recovery Score Calculation
-> Recovery Priority Recommendation
-> Real-Time Dashboard
```

The motivation is that e-waste contains recoverable materials such as copper,
aluminum, gold, silver, palladium, lithium, cobalt, nickel, and rare-earth
metals. If a camera-based AI system can quickly identify components, it can help
prioritize which items should be recovered first.

## 2. What Was Already Present

Before the most recent improvements, the project already had:

- A basic image classification pipeline using TensorFlow/Keras.
- A YOLO live-detection script.
- A `material_values.json` file with material information for several e-waste
  categories.
- A fallback YOLO demo mode using the generic YOLO11 model.
- A basic live camera detection window using OpenCV.
- A dataset under `data/images/` with many e-waste-related image folders.

The earlier project could demonstrate the general idea, but it needed better
research structure, clearer dataset analysis, camera troubleshooting tools, an
expanded component taxonomy, and better documentation.

## 3. What Has Been Added

### Expanded Component Taxonomy

The project now supports a broader and more research-relevant component list:

- `PCB`
- `cable`
- `battery`
- `ram`
- `heat_sink`
- `cpu`
- `gpu`
- `ssd`
- `hard_drive`
- `smartphone`
- `tablet`
- `display_screen`
- `keyboard`
- `mouse`
- `printer`
- `charger_power_adapter`
- `power_supply`
- `laptop_motherboard`
- `connector`
- `capacitor`
- `transformer`
- `fan`
- `electric_motor`
- `speaker`
- `optical_drive`
- `metal_component`
- `plastic_housing`

This is defined in:

```text
ewaste_research/taxonomy.py
```

Why this matters:

The older labels were too broad. A real e-waste recovery system should separate
high-value components such as CPUs, GPUs, RAM, SSDs, and laptop motherboards
because they have different material compositions and recovery priorities.

### Expanded Material Database

`material_values.json` was expanded with richer entries for:

- RAM modules
- heat sinks
- CPUs
- GPUs
- SSDs
- hard drives
- power supplies
- laptop motherboards
- batteries
- PCBs
- cables
- mixed metal components
- plastic housings

Each rich entry includes:

- Display name
- Estimated value in dollars per kilogram
- Recovery priority
- Notes
- Material composition percentages

Example fields:

```json
{
  "display_name": "CPU / Processor",
  "recovery_priority": "critical",
  "estimated_value_usd_per_kg": 42.0,
  "materials": [
    { "material": "Copper", "percent": 20.0 },
    { "material": "Gold", "percent": 0.08 }
  ]
}
```

### Recovery Score Calculation

A new recovery score was added in:

```text
ewaste_research/materials.py
```

The score combines:

- Estimated economic value per kilogram
- Copper content
- Precious metal content

The score is normalized from 0 to 100.

Current weighting:

```text
50% estimated value
30% copper content
20% precious metal content
```

Why this matters:

Instead of only saying "this is a PCB," the system can now estimate how valuable
or important the detected item is for recovery.

### Real-Time Dashboard

The live detector now has a dashboard panel that displays:

- Detected component
- Confidence score
- Estimated value
- Copper percentage
- Precious metal percentage
- Recovery score
- Recommended priority
- Top material composition values

The dashboard rendering code is in:

```text
ewaste_research/dashboard.py
```

The main live detector is:

```text
detect_ewaste.py
```

### Dataset Analysis Tool

A new dataset analysis script was added:

```text
analyze_dataset.py
```

It generates:

- Class distribution counts
- Underrepresented class detection
- CSV tables
- Publication-quality bar graphs
- JSON summary files

Generated report locations:

```text
artifacts/reports/dataset_analysis/
artifacts/reports/dataset_analysis_augmented/
```

Current local dataset analysis:

```text
Original dataset size: 23,960 images
First augmented dataset size: 24,660 images
Expanded augmented dataset size: 27,380 images
```

The original dataset had one underrepresented canonical class:

```text
heat_sink: 800 images
```

After augmentation:

```text
heat_sink: 1,500 images
Mobile/smartphone: 1,500 images
Mouse: 1,500 images
PCB: 1,500 images
Television/display_screen: 1,500 images
Underrepresented canonical classes: none at the 900-image threshold
```

### Data Augmentation Tool

A targeted augmentation script was added:

```text
augment_low_frequency.py
```

It creates augmented images using mild, realistic transformations:

- Horizontal flip
- Small rotation
- Brightness adjustment
- Contrast adjustment
- Color adjustment
- Autocontrast

This was used to increase `heat-sink`, `Mobile`, `Mouse`, `PCB`, and
`Television` to 1,500 images each in the expanded augmented dataset.

Output dataset:

```text
data/images_augmented/
```

Why this matters:

Class imbalance can cause the model to perform poorly on low-frequency classes.
Augmentation helps the model see more visual variation without manually
collecting hundreds of new images.

### YOLO Evaluation Tool

A YOLO evaluation script was added:

```text
evaluate_yolo.py
```

It is designed to report:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Mean inference latency
- Median inference latency
- Estimated FPS

Important limitation:

This requires a trained YOLO model at:

```text
artifacts/models/best.pt
```

That file is not currently present. The project currently has:

```text
artifacts/models/best_image_classifier.keras
```

That is the older image classifier, not the YOLO object detector.

### Recovery Priority Ranking

A recovery ranking tool was added:

```text
rank_recovery.py
```

It creates:

```text
artifacts/reports/recovery_priority_table.csv
```

Current top recovery targets:

```text
1. CPU / Processor
2. Connector / Port
3. Laptop Motherboard
4. Smartphone
5. RAM Module
6. Graphics Card / GPU
7. Printed Circuit Board
8. Cable / Wiring
9. Tablet
10. Solid State Drive (SSD)
```

This makes the project recovery-first: the system is not only asking "what is
the object?" It is asking "which detected object should be recovered first?"

### Arducam / USB Camera Support

Several camera support tools were added:

```text
camera_probe.py
arducam_viewer.py
run_arducam_viewer.ps1
run_ewaste_dashboard.ps1
ewaste_research/camera.py
```

The camera was verified as:

```text
Device name: USB Camera
Working source index: 0
Working backend: default
Resolution: 1280x720
FPS: 30
```

A successful snapshot was saved at:

```text
artifacts/reports/camera_probe/camera_0_default_snapshot.jpg
```

The project can now:

- Probe connected cameras
- Save test snapshots
- Open a plain camera viewer without YOLO
- Launch the e-waste dashboard with the known working camera settings
- Auto-scan camera indexes and OpenCV backends

## 4. How To Run The Demo

### Test The Camera Only

Use this first if the camera is acting weird:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_arducam_viewer.ps1
```

Close the window with `Q` or `Esc`.

### Run The E-Waste Dashboard

Use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_ewaste_dashboard.ps1
```

Or directly:

```powershell
python detect_ewaste.py --source 0 --backend default --width 1280 --height 720 --conf 0.25
```

Close the dashboard with `Q` or `Esc`.

Important:

Only one app can use the camera at a time. Close the Windows Camera app, Zoom,
Teams, browser camera tabs, `arducam_viewer.py`, and old detector windows before
running the dashboard.

## 5. Current Honest System Status

### Working Now

- The Arducam / USB camera can be detected by Python.
- The system can capture frames from the camera.
- The live OpenCV window can run.
- The material database lookup works.
- The recovery score calculation works.
- The dashboard rendering code works.
- Dataset analysis works.
- Dataset augmentation works.
- Poster-style graphs and CSV tables are generated.
- Camera probing and troubleshooting tools are available.

### Partially Working

- The YOLO live demo can run in fallback mode using generic YOLO11 weights.
- Fallback mode can detect general objects such as phones, laptops, keyboards,
  and map them to e-waste categories.

### Not Finished Yet

- A real custom-trained YOLO e-waste detector is still needed.
- The file `artifacts/models/best.pt` is missing.
- The current dataset folders are mostly classification-style folders, not fully
  YOLO object-detection labels.
- More bounding-box annotations are needed for CPUs, GPUs, SSDs, RAM, PSUs,
  cables, laptop motherboards, and other target components.
- Evaluation metrics cannot be finalized until the trained YOLO model exists.

## 6. What Still Needs To Be Done

### Step 1: Build A True YOLO Dataset

YOLO requires images and bounding-box label files.

Expected structure:

```text
data/yolo_raw/
  image_001.jpg
  image_001.txt
  image_002.jpg
  image_002.txt
```

Each `.txt` file contains YOLO bounding box labels:

```text
class_id x_center y_center width height
```

All coordinates are normalized between 0 and 1.

### Step 2: Label More Component Images

Use Label Studio or Roboflow to draw boxes around:

- CPUs
- GPUs
- SSDs
- HDDs
- RAM
- cables
- batteries
- PCBs
- heat sinks
- power supplies
- laptop motherboards

The most important next research task is high-quality labeling.

### Step 3: Prepare YOLO Dataset

Run:

```powershell
python prepare_yolo_data.py --source-dir data/yolo_raw
```

This creates:

```text
dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  dataset.yaml
```

### Step 4: Train YOLO

Use:

```text
train_yolo_colab.ipynb
```

Train on Google Colab with a GPU.

The trained model output should be:

```text
best.pt
```

Move it into:

```text
artifacts/models/best.pt
```

### Step 5: Evaluate The Model

After `best.pt` exists, run:

```powershell
python evaluate_yolo.py --model artifacts/models/best.pt --data-yaml dataset/dataset.yaml
```

This produces precision, recall, mAP, and latency metrics.

### Step 6: Run Final Real-Time Demo

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_ewaste_dashboard.ps1
```

At that point, the system should perform true e-waste component detection instead
of fallback generic-object detection.

## 7. What To Tell Professors

Short version:

> This project combines real-time object detection with a material recovery
> database. The camera detects e-waste components, the detected class is used to
> look up estimated material composition, and the system calculates a recovery
> score based on economic value, copper content, and precious metal content.

More detailed version:

> The project currently has a working software pipeline for camera capture,
> detection display, material lookup, recovery scoring, dataset analysis,
> class-balancing augmentation, and evaluation reporting. The remaining research
> task is to complete high-quality YOLO bounding-box annotations and train the
> final custom detector.

Honest limitation:

> The current live demo can run in fallback mode using a generic YOLO model, but
> the final custom e-waste YOLO detector still requires a trained `best.pt` file.
> The infrastructure is ready; the main remaining bottleneck is labeled
> detection data.

Strong research contribution:

> The project is not only detecting objects. It connects computer vision output
> to recovery decision-making by estimating material value, copper content,
> precious-metal content, and recovery priority.

## 8. Files To Know

| File | Purpose |
| --- | --- |
| `detect_ewaste.py` | Main live camera detection and dashboard script |
| `arducam_viewer.py` | Camera-only viewer for troubleshooting |
| `camera_probe.py` | Finds working camera indexes and saves snapshots |
| `material_values.json` | Material composition and value database |
| `ewaste_research/materials.py` | Recovery score calculation |
| `ewaste_research/taxonomy.py` | Canonical component taxonomy |
| `ewaste_research/dashboard.py` | Dashboard rendering |
| `ewaste_research/camera.py` | Camera auto-detection helpers |
| `analyze_dataset.py` | Dataset class-distribution analysis |
| `augment_low_frequency.py` | Targeted data augmentation |
| `evaluate_yolo.py` | YOLO precision, recall, mAP, and latency evaluation |
| `prepare_yolo_data.py` | Converts raw YOLO labels into YOLO training format |
| `train_yolo_colab.ipynb` | Colab notebook for YOLO training |
| `run_arducam_viewer.ps1` | One-command camera viewer |
| `run_ewaste_dashboard.ps1` | One-command dashboard launcher |

## 9. Poster / Paper Figures Already Generated

Dataset analysis outputs:

```text
artifacts/reports/dataset_analysis/class_distribution.png
artifacts/reports/dataset_analysis/class_distribution.csv
artifacts/reports/dataset_analysis/dataset_summary.json
```

Augmented dataset analysis outputs:

```text
artifacts/reports/dataset_analysis_augmented/class_distribution.png
artifacts/reports/dataset_analysis_augmented/class_distribution.csv
artifacts/reports/dataset_analysis_augmented/dataset_summary.json
```

Expanded augmented dataset analysis outputs:

```text
artifacts/reports/dataset_analysis_augmented_expanded/class_distribution.png
artifacts/reports/dataset_analysis_augmented_expanded/class_distribution.csv
artifacts/reports/dataset_analysis_augmented_expanded/dataset_summary.json
```

Camera probe output:

```text
artifacts/reports/camera_probe/camera_probe.json
artifacts/reports/camera_probe/camera_0_default_snapshot.jpg
```

Recovery ranking output:

```text
artifacts/reports/recovery_priority_table.csv
```

## 10. Suggested Next Milestones

1. Finalize the component taxonomy.
2. Collect or curate more component-specific images.
3. Label bounding boxes for the expanded classes.
4. Build the YOLO dataset with `prepare_yolo_data.py`.
5. Train YOLO in Colab.
6. Save `best.pt` into `artifacts/models/`.
7. Run `evaluate_yolo.py`.
8. Capture final dashboard screenshots or video.
9. Build poster tables from the generated CSV/JSON outputs.
10. Present the system as an end-to-end AI-assisted recovery prototype.
