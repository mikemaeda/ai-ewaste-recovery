# Final Poster Draft

## Title

AI-Assisted E-Waste Recovery: Connecting Computer Vision to Material-Recovery Decisions

## Authors / Affiliation

Mike Maeda  
Undergraduate Research Project  
Department / Program: Computer Science / Engineering / Sustainability

## Research Question

Can a computer-vision system identify e-waste components and prioritize them for material recovery using an interpretable recovery scoring model?

## Introduction

Electronic waste contains recoverable copper, gold, silver, palladium, lithium, cobalt, nickel, aluminum, and rare-earth materials, but sorting remains a practical bottleneck.

Most image-classification systems answer only one question: "What object is this?" This project asks a second question: "Which detected component should be recovered first?"

This work connects computer-vision predictions to a material-recovery database so that detected components can be ranked by estimated economic value, copper content, and precious-metal content.

## Methods

The project began with a MobileNetV2 transfer-learning classifier trained on 23,960 labeled e-waste images across 18 classes. Images were split 80/20 for training and validation, and the model was trained for 10 epochs.

Dataset analysis identified class imbalance. Targeted augmentation using flips, small rotations, brightness, contrast, color adjustment, and autocontrast expanded the working dataset to 27,380 images. In the expanded dataset, all canonical classes are above the 900-image low-frequency threshold.

The material recovery layer uses a curated `material_values.json` database containing estimated material composition, copper percentage, precious-metal percentage, estimated value per kilogram, and recovery priority.

Recovery score:

```text
score = 0.50 * value + 0.30 * copper + 0.20 * precious metals
```

Each term is normalized to a 0-100 scale before weighting.

System pipeline:

```text
Camera / Image
-> Computer Vision Model
-> Component Class
-> Material Database Lookup
-> Recovery Score
-> Priority Recommendation
```

## Results

The MobileNetV2 classifier reached 90.7% validation accuracy after 10 epochs.

The expanded augmented dataset contains 27,380 images across 13 canonical dataset classes, with no underrepresented class below the 900-image threshold.

Recovery scores were calculated for 27 e-waste components. The highest-ranked recovery targets were:

| Rank | Component | Score | Estimated Value | Copper | Precious Metals | Priority |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | CPU / Processor | 79.2 | $42.00/kg | 20.0% | 0.285% | critical |
| 2 | Connector / Port | 75.2 | $36.00/kg | 55.0% | 0.060% | critical |
| 3 | Laptop Motherboard | 66.3 | $32.00/kg | 21.0% | 0.208% | critical |
| 4 | Smartphone | 63.5 | $30.00/kg | 13.0% | 0.365% | critical |
| 5 | RAM Module | 57.5 | $28.00/kg | 18.0% | 0.177% | high |
| 6 | Graphics Card / GPU | 53.6 | $24.00/kg | 24.0% | 0.157% | critical |
| 7 | Printed Circuit Board | 42.6 | $18.00/kg | 20.0% | 0.136% | high |
| 8 | Cable / Wiring | 36.9 | $5.50/kg | 65.0% | 0.000% | high |
| 9 | Tablet | 35.6 | $18.00/kg | 10.0% | 0.106% | high |
| 10 | Solid State Drive (SSD) | 33.0 | $14.00/kg | 14.0% | 0.113% | high |

The ranking shows that high-grade boards and small components such as CPUs, connectors, motherboards, smartphones, RAM, and GPUs dominate recovery priority. Copper-rich items such as cables remain important even when their estimated dollar value is lower.

## Discussion

The main contribution is an interpretable bridge between computer vision and recovery decision support. The system does not stop at classification; it converts predicted component labels into recovery scores that can be audited and explained.

Because the scoring formula is transparent, the system is easier to explain to recyclers, researchers, and reviewers than a black-box recommendation model.

Important limitations remain. The custom YOLO object detector is not fully trained yet, so the current live-camera detector can run in fallback mode but does not yet report final object-detection metrics. The classifier taxonomy and the expanded material-recovery taxonomy are also not fully unified. Material composition estimates currently rely on published literature values rather than direct spectroscopic measurement.

## Conclusion

This project demonstrates that AI can support e-waste recovery by connecting visual recognition to material-value estimation. The prototype identifies components, estimates recoverable materials, calculates a recovery score, and ranks components by priority.

The strongest result is the recovery-first pipeline: computer vision output becomes actionable decision support for recycling and material recovery.

## Future Work

- Label bounding boxes for the expanded component taxonomy.
- Train and evaluate the custom YOLO detector.
- Report object-detection precision, recall, mAP, latency, and FPS.
- Unify the classifier classes with the full 27-component recovery taxonomy.
- Validate material composition estimates against measured samples.
- Deploy the real-time Arducam dashboard as a demonstration system.

## Required Figures

Use these figures no matter what:

- `fig7_classifier_headline.png` - 90.7% accuracy result.
- `fig1_recovery_ranking.png` - signature recovery ranking visual.
- `fig9_material_table.png` - top-10 recovery target table.
- `fig8_pipeline.png` - end-to-end system pipeline.

Use these if there is space:

- `fig2_score_composition.png` - score decomposition.
- `fig3_value_vs_score.png` - economic value vs. recovery score.
- `fig6_dataset_distribution.png` - expanded balanced dataset.
- `fig4_copper_content.png` or `fig5_precious_metals.png` - material detail.

