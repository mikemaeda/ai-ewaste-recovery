# Poster Content Guide

**AI-Assisted E-Waste Recovery Through Computer Vision and Material Composition Estimation**

All figures are in `artifacts/reports/poster/` at 300 DPI (print-ready). Every
number is generated from the project's own data — no invented values.

Regenerate any time with: `python make_poster_figures.py`

---

## Figure inventory (what each PNG is and where it goes)

| File | Figure | Poster section |
|------|--------|----------------|
| `fig8_pipeline.png` | End-to-end system pipeline | Methods |
| `fig6_dataset_distribution.png` | Training dataset after augmentation | Methods / Results |
| `fig7_classifier_headline.png` | 90.7% accuracy hero + stat tiles | Results (banner) |
| `fig1_recovery_ranking.png` | **Recovery priority score, all 27 components** | Results (signature) |
| `fig2_score_composition.png` | How the recovery score is built | Results |
| `fig9_material_table.png` | Top-10 recovery targets table | Results |
| `fig4_copper_content.png` | Recoverable copper by component | Results |
| `fig5_precious_metals.png` | Gold/silver/palladium content | Results |
| `fig3_value_vs_score.png` | Economic value vs. recovery score | Results / Discussion |

---

## Recommended layout (3 columns) & space budget

```
+----------------------------------------------------------+
|                 TITLE + name / affiliation               |
+-------------------+------------------+-------------------+
| INTRODUCTION 12%  |  RESULTS  ~45%   |  DISCUSSION 15%   |
|  (short text)     |  fig7 banner     |  (text)           |
|                   |  fig1 signature  |                   |
| METHODS 20%       |  fig9 table      |  CONCLUSION 8%    |
|  fig8 pipeline    |  fig2 + fig3     |                   |
|  fig6 dataset     |  fig4 + fig5     |  References/QR    |
+-------------------+------------------+-------------------+
```

- **Left column:** Introduction (top), then Methods (pipeline + dataset).
- **Center column:** Results — the largest zone. Lead with the 90.7% banner,
  then the recovery-ranking figure (the single most important visual), then the
  table, then the supporting composition figures.
- **Right column:** Discussion + Conclusion + limitations + QR/references.

---

## Section-by-section content

### 1. Introduction (~12%, mostly text)
- E-waste is the fastest-growing solid waste stream; most valuable metals
  (copper, gold, silver, palladium, cobalt) stay locked in discarded devices.
- Manual sorting is slow and expensive.
- **Gap this project fills:** classification alone says *what* an object is — it
  does not say *what it is worth recovering*. This project links vision output to
  a material-recovery decision.
- One-line thesis (put this big): *"We connect computer-vision predictions to
  material-recovery decisions."*

### 2. Methods (~20%)
- **Figure:** `fig8_pipeline.png` — the six-stage pipeline.
- **Figure:** `fig6_dataset_distribution.png` — dataset scale & class balancing.
- Text bullets: MobileNetV2 transfer-learning classifier; 18 classes;
  23,960 labeled images; 80/20 train/val split; 10 epochs. Class-balancing
  augmentation (flip, small rotation, brightness/contrast/color, autocontrast)
  grew the corpus to 27,380 and eliminated under-represented classes.
- Tech stack: Python, TensorFlow/Keras, Ultralytics YOLO, OpenCV, NumPy,
  Matplotlib.
- Recovery score is a transparent weighted formula (state it verbatim):
  `score = 0.50·value + 0.30·copper + 0.20·precious`, each term normalized 0–100.

### 3. Results (~45%, dominant)
- **Banner:** `fig7_classifier_headline.png` — 90.7% validation accuracy.
- **Signature:** `fig1_recovery_ranking.png` — priority score for all 27
  components, colored by priority tier.
- **Table:** `fig9_material_table.png` — top-10 targets with value, copper,
  precious %, score, priority.
- **Algorithm figure:** `fig2_score_composition.png` — shows the three weighted
  contributions stacked, so the reader sees *why* CPUs and connectors win.
- **Composition:** `fig4_copper_content.png` and `fig5_precious_metals.png`.
- **Analysis:** `fig3_value_vs_score.png` — value vs. score, bubble = copper %;
  makes the point that copper-rich, low-dollar parts (cable, transformer) still
  score highly.

### 4. Discussion (~15%, text)
- **Strengths:** moves beyond classification to decision support; transparent,
  auditable scoring formula; balanced dataset; end-to-end software already runs.
- **Limitations (be honest):**
  - The custom YOLO e-waste detector is **not yet trained** — `best.pt` does not
    exist. The live demo currently runs in fallback mode on generic YOLO11
    weights, mapping general objects to e-waste categories.
  - The classifier (18 classes) and the material/recovery taxonomy (27
    components) are **not yet unified**; bridging them is future work.
  - *"Material composition estimates currently rely on published literature
    values rather than direct spectroscopic measurement."* (Use this sentence
    verbatim — it is precise and academic.)
  - No bounding-box annotations yet for most high-value components, so detection
    metrics (mAP, precision/recall) are not reported.

### 5. Conclusion (~8%)
- AI can identify e-waste components and prioritize them for recovery.
- Material estimation converts classification into decision support.
- Future work: unify taxonomies, annotate bounding boxes, train and evaluate the
  custom YOLO detector, and deploy the real-time camera dashboard.

---

## Paste-ready figure captions

- **Fig 1.** Recovery priority score for 27 e-waste components. Score combines
  economic value (0.50), copper content (0.30), and precious-metal content
  (0.20), normalized to 0–100. CPUs, connectors, and laptop motherboards rank
  highest.
- **Fig 2.** Decomposition of the recovery score for the top 12 components into
  its three weighted contributions, illustrating why high-value boards dominate.
- **Fig 3.** Estimated economic value vs. recovery score; bubble size encodes
  recoverable copper. Copper-rich but low-value parts (cable, transformer) still
  earn meaningful scores.
- **Fig 4.** Recoverable copper content by component (% by mass).
- **Fig 5.** Gold, silver, and palladium content of high-value components
  (% by mass); note the sub-0.2% scale.
- **Fig 6.** Training dataset after class-balancing augmentation: 27,380 images
  across 13 component classes, all above the 900-image low-frequency threshold.
- **Fig 7.** MobileNetV2 classifier: 90.7% validation accuracy over 10 epochs on
  an 80/20 split of 23,960 labeled images (18 classes).
- **Fig 8.** End-to-end recovery pipeline from camera capture to priority
  recommendation.
- **Table 1 (Fig 9).** Top 10 recovery targets with estimated value, copper %,
  precious-metal %, recovery score, and recommended priority.

---

## Honest data notes (so you never get caught off guard)
- **90.7%** is the real `validation_accuracy` from `artifacts/reports/evaluation.json`
  (0.9065). Use "90.7%", not "91.2%".
- The classifier's 18 classes come from the training run; the recovery/material
  system uses a broader 27-component taxonomy. Present them as two layers of the
  same system, not one model.
- Recovery scores, copper %, and precious % all come straight from
  `artifacts/reports/recovery_priority_table.csv`, which is computed from
  `material_values.json` by `ewaste_research/materials.py`.
