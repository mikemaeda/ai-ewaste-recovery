# Data

Raw image datasets are not committed to this repository because they are large and may carry external licensing requirements.

Expected classifier layout:

```text
data/images/
  Class_A/
    image_001.jpg
  Class_B/
    image_001.jpg
```

Expected YOLO raw-label layout:

```text
data/yolo_raw/
  image_001.jpg
  image_001.txt
```

See `DATASET_SOURCES.md`, `DATASET_ACQUISITION_PLAN.md`, and `README_YOLO.md` for dataset source notes and preparation steps.
