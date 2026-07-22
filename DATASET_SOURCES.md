# Dataset Expansion Sources

Use this file as a research log for expanding the e-waste detection dataset. I
cannot download Kaggle data without your Kaggle API token, but the project now
has scripts that can analyze and augment whatever you place in `data/images/`
or YOLO format under `dataset/`.

## Good search targets

Kaggle, Hugging Face, and Roboflow searches to try:

- `e waste image dataset`
- `electronic waste classification`
- `e-waste object detection`
- `electronic waste image classification`
- `printed circuit board object detection`
- `pcb components detection`
- `computer parts detection`
- `hard drive SSD GPU CPU dataset`
- `CPU GPU RAM SSD object detection`
- `charger adapter electronics dataset`
- `laptop motherboard dataset`
- `recycling waste computer components`

See also:

```text
external_dataset_candidates.csv
DATASET_ACQUISITION_PLAN.md
```

## Candidate dataset types

Prioritize these because they match the expanded taxonomy:

| Target class | Useful source labels to merge |
| --- | --- |
| `PCB` | PCB, circuit board, motherboard, electronics board |
| `cable` | wire, cable, charger cable, copper wire |
| `battery` | lithium battery, phone battery, laptop battery |
| `ram` | RAM, memory module, DIMM |
| `heat_sink` | heatsink, heat sink, cooler |
| `cpu` | CPU, processor, microchip, IC |
| `gpu` | graphics card, video card, GPU |
| `ssd` | SSD, solid state drive, M.2 drive |
| `hard_drive` | HDD, hard disk drive |
| `smartphone` | phone, mobile phone, cell phone |
| `tablet` | tablet, iPad, mobile device |
| `display_screen` | TV, monitor, screen, LCD |
| `keyboard` | keyboard, computer keyboard |
| `mouse` | mouse, computer mouse |
| `printer` | printer, inkjet, laser printer |
| `charger_power_adapter` | charger, adapter, power brick |
| `power_supply` | PSU, computer power supply |
| `laptop_motherboard` | motherboard, laptop motherboard, mainboard |
| `connector` | port, connector, socket, gold contact |
| `capacitor` | capacitor, electrolytic capacitor |
| `transformer` | transformer, inductor, coil |
| `fan` | fan, cooling fan |
| `electric_motor` | motor, compressor, appliance motor |
| `speaker` | speaker, audio driver |
| `optical_drive` | CD drive, DVD drive, optical drive |

## Kaggle API setup

1. Create a Kaggle API token from your Kaggle account settings.
2. Put `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`.
3. Install the Kaggle CLI:

```powershell
python -m pip install kaggle
```

4. Download into `data/external/<dataset-name>/`, then curate/rename classes
   into `data/images/` or label boxes into `data/yolo_raw/`.

## Required quality checks

Before training, run:

```powershell
python analyze_dataset.py --data-dir data/images
python augment_low_frequency.py --source-dir data/images --output-dir data/images_augmented --target-count 1500
```

For YOLO labels:

```powershell
python analyze_dataset.py --yolo-dir dataset
python prepare_yolo_data.py --source-dir data/yolo_raw
```

Keep a spreadsheet or notes file with each external dataset's title, URL,
license, number of usable images, and classes retained. That provenance matters
for a poster, paper, or publication.
