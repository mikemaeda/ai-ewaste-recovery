# e-waste AI — Material Recovery (YOLO Live Detection)

This is the **object-detection** part of the project. It opens a live window on
your screen and draws boxes around e-waste objects (PCBs, cables, batteries,
hard drives, metal components, plastic housings), labels them, and prints what
valuable materials each one contains.

> This sits **alongside** the existing image-classifier (`ewaste_research/cli/train_classifier.py`,
> `ewaste_research/cli/predict_classifier.py`, `ewaste_research/cli/prepare_classifier_data.py`). None of those files were changed. The YOLO
> files all have `yolo` in their name or are listed below.

## The 6 detection classes

`PCB`, `cable`, `battery`, `ram`, `heat_sink`, `cpu`, `gpu`, `ssd`,
`hard_drive`, `power_supply`, `laptop_motherboard`, `metal_component`,
`plastic_housing`

Material composition, recovery priority, and scrap value for each are stored in
`data/material-values.json`.

The live detector also calculates a **recovery score** from estimated value,
copper content, and precious-metal content, then shows the result in a real-time
dashboard side panel.

---

## Quick start — see the window in 2 minutes (no training needed)

You do **not** need a trained model to see it work. There is a fallback mode
that uses a pretrained model and maps everyday objects (phone, laptop, keyboard…)
onto e-waste categories.

1. Open this project folder in **VS Code**.
2. Open a **PowerShell** terminal (Terminal menu > New Terminal).
3. Install everything (one time):

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts/windows/setup-yolo.ps1
   ```

4. Run the live detector on your webcam:

   ```powershell
   python -m ewaste_research.cli.run_detector --source auto --backend auto --width 1280 --height 720 --conf 0.25
   ```

   A window pops up. Point your webcam at a phone, laptop, or keyboard and you'll
   see boxes + labels, and material readouts in the terminal.
   **Press `Q` to quit.**

   Or run it on a single photo instead of the webcam:

   ```powershell
   python -m ewaste_research.cli.run_detector --source path\to\photo.jpg
   ```

That's the whole "one command and a window appears" goal. Everything below is
how to train your **own** accurate model.

---

## Full pipeline (to detect real e-waste accurately)

### Step 0 - Analyze and expand the dataset

Generate poster-ready class-distribution outputs:

```powershell
python -m ewaste_research.cli.analyze_dataset --data-dir data/images
```

Balance underrepresented folder classes with camera-realistic augmentation:

```powershell
python -m ewaste_research.cli.augment_dataset --source-dir data/images --output-dir data/images_augmented --target-count 1500
```

For external datasets, see `docs/dataset-sources.md`. Once you find Kaggle dataset
slugs, this helper prints the download commands:

```powershell
python -m ewaste_research.cli.download_dataset --kaggle-slug owner/dataset-name
```

### Step 1 — Install

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/windows/setup-yolo.ps1
```

This installs Ultralytics (YOLO11), OpenCV, and the rest from
`requirements_yolo.txt`.

### Step 2 — Label your images with Label Studio

YOLO needs images **with boxes drawn around the objects**. You already have
thousands of e-waste photos in `data/images/` (PCB, Battery, Keyboard, Laptop,
Mouse, Microchip-IC, …). Use those as your source images.

1. Install and start Label Studio:

   ```powershell
   python -m pip install label-studio
   label-studio start
   ```

   It opens in your browser at `http://localhost:8080`.

2. Create an account (local, just click through), then **Create Project**.
3. **Data Import**: drag in images from `data/images/` — pick a mix of the
   folders that match our 6 classes (e.g. `PCB`, `Battery`, plus phones/laptops
   that contain hard drives and plastic housings).
4. **Labeling Setup** > choose **Object Detection with Bounding Boxes**.
5. Add exactly these 6 labels (spelling matters):
   `PCB`, `cable`, `battery`, `hard_drive`, `metal_component`, `plastic_housing`
6. Draw a box around each object in each image and pick its label. Save/Submit.
7. When done: **Export** > format **YOLO**. You get a zip with an `images/`
   folder and a `labels/` folder of `.txt` files.
8. Put all those images **and** their `.txt` files together into one folder:

   ```
   data/yolo_raw/
   ```

   (image and its label share the same name, e.g. `pcb_001.jpg` + `pcb_001.txt`)

> Aim for at least ~30–50 labeled images per class to start. More is better.

### Step 3 — Build the YOLO dataset folder

```powershell
python -m ewaste_research.cli.prepare_yolo_data --source-dir data/yolo_raw
```

This resizes everything to 640×640, splits into train/val, and writes:

```
dataset/
  images/train/   images/val/
  labels/train/   labels/val/
  dataset.yaml
```

### Step 4 — Train on Google Colab (free GPU)

1. Zip the `dataset` folder → `dataset.zip` (right-click > Send to >
   Compressed (zipped) folder).
2. Go to [Google Colab](https://colab.research.google.com/) and upload
   `notebooks/yolo-training-colab.ipynb` (File > Upload notebook).
3. Set **Runtime > Change runtime type > T4 GPU**.
4. Run the cells top to bottom. The notebook mounts Drive, installs Ultralytics,
   takes your `dataset.zip`, trains YOLO11, and downloads **`best.pt`**.

### Step 5 — Use your trained model locally

1. Move the downloaded `best.pt` into:

   ```
   artifacts/models/best.pt
   ```

2. Run the live window with your model:

   ```powershell
   python -m ewaste_research.cli.run_detector --model artifacts/models/best.pt --source 0
   ```

   `ewaste_research/cli/run_detector.py` automatically uses `artifacts/models/best.pt` if it exists,
   so once the file is there you can just run `python -m ewaste_research.cli.run_detector`.

---

## Command reference

```powershell
# Find the connected Arducam/webcam index and save test snapshots
python -m ewaste_research.cli.probe_camera

# Webcam, default model location (trained if present, else fallback)
python -m ewaste_research.cli.run_detector --source auto --backend auto --width 1280 --height 720

# Specific webcam index (try 1 if 0 is wrong)
python -m ewaste_research.cli.run_detector --source 1

# A single image
python -m ewaste_research.cli.run_detector --source photo.jpg

# Explicit trained model + lower confidence threshold
python -m ewaste_research.cli.run_detector --model artifacts/models/best.pt --conf 0.25
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--source` | Webcam index, `auto`, or image/video path | `auto` |
| `--backend` | OpenCV webcam backend: `auto`, `default`, `dshow`, or `msmf` | `auto` |
| `--width` / `--height` | Requested webcam resolution | `1280` / `720` |
| `--model` | Path to trained `.pt` weights | `artifacts/models/best.pt` |
| `--conf` | Minimum confidence to show a box (0–1) | `0.35` |
| `--values` | Path to material data | `data/material-values.json` |

---

## YOLO files added (nothing existing was overwritten)

| File | Purpose |
|------|---------|
| `requirements_yolo.txt` | YOLO dependencies (separate from `requirements.txt`) |
| `scripts/windows/setup-yolo.ps1` | One-command installer |
| `ewaste_research/cli/prepare_yolo_data.py` | Builds the YOLO `dataset/` layout + `dataset.yaml` |
| `ewaste_research/cli/train_yolo.py` | Local YOLO training entry point for prepared detection datasets |
| `notebooks/yolo-training-colab.ipynb` | Colab notebook to train YOLO11 |
| `ewaste_research/cli/run_detector.py` | **The live on-screen detection window** |
| `ewaste_research/cli/probe_camera.py` | Finds working camera indexes and saves test snapshots |
| `ewaste_research/cli/view_camera.py` | Opens the camera feed without YOLO for quick hardware testing |
| `scripts/windows/run-camera-viewer.ps1` | One-command Arducam viewer launcher |
| `scripts/windows/run-detector.ps1` | One-command live dashboard launcher |
| `ewaste_research/cli/analyze_dataset.py` | Class distribution, underrepresented classes, poster graph + CSV |
| `ewaste_research/cli/augment_dataset.py` | Targeted augmentation for low-frequency classes |
| `ewaste_research/cli/evaluate_yolo.py` | Precision, recall, mAP, and inference latency reporting |
| `ewaste_research/cli/rank_recovery.py` | Ranks classes by recovery score and exports a poster-ready CSV |
| `ewaste_research/` | Shared taxonomy, material scoring, plotting, and dashboard code |
| `docs/dataset-sources.md` | Dataset expansion notes for Kaggle/Roboflow searches |
| `docs/dataset-acquisition.md` | Online data search, labeling, and training workflow |
| `data/external-dataset-candidates.csv` | Candidate source log for Kaggle, Hugging Face, Roboflow, and papers |
| `data/material-values.json` | Extended with rich data for the expanded component taxonomy |
| `docs/yolo-workflow.md` | This file |

---

## Troubleshooting

- **Window doesn't open / black webcam**: another app may be using the camera.
  Close it, or try `--source 1`.
- **`Could not open video source`**: no webcam — run on an image instead:
  `python -m ewaste_research.cli.run_detector --source somephoto.jpg`.
- **Install is slow**: that's PyTorch downloading (hundreds of MB). Let it finish.
- **Fallback labels look wrong** (e.g. a phone tagged `PCB`): expected — fallback
  maps everyday COCO objects to e-waste categories as a placeholder. Train your
  own model (Steps 2–5) for real accuracy.
- **`python` not found**: install Python from python.org and tick "Add to PATH".
```
