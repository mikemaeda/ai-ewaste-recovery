# Three Poster Content Drafts

These are concise poster-ready drafts based on the same project data. Use Draft
3 for the final poster unless your professor specifically wants a more technical
or sustainability-heavy version.

## Draft 1 - Technical / Computer Science Focus

### Short Title Option

Vision-to-Value: Linking CNN Predictions to E-Waste Recovery Scores

### Research Question / Objective

Can a computer-vision system move beyond classifying e-waste to ranking
components by recovery value using a transparent scoring model?

### Introduction

- Image classifiers answer "what is this?" but not "is it worth recovering?"
- This project adds a decision layer: prediction -> material lookup -> recovery score.
- Goal: a lightweight and auditable pipeline for e-waste recovery support.

### Methods

- MobileNetV2 transfer learning.
- 23,960 labeled images, 18 classes, 80/20 split, 10 epochs.
- Class-balancing augmentation expanded the working dataset to 27,380 images.
- Recovery score:

```text
score = 0.50 * value + 0.30 * copper + 0.20 * precious metals
```

- Stack: Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Ultralytics YOLO.

### Results

- Classifier reached 90.7% validation accuracy.
- Recovery score computed for 27 components.
- Top recovery targets: CPU, connector, laptop motherboard, smartphone, RAM, GPU.
- Score decomposition shows why high-grade boards and connectors rank highest.
- Dataset analysis shows no underrepresented canonical classes below the 900-image threshold after augmentation.

### Discussion

- The recovery formula is transparent and inspectable.
- Copper-rich, lower-dollar parts still receive meaningful priority.
- The custom YOLO detector remains future work because final bounding-box annotations are still needed.

### Conclusion

The project turns classification into recovery-priority ranking using an
interpretable computer-vision pipeline.

### Future Work

- Label bounding boxes.
- Train YOLO and report mAP, precision, recall, latency, and FPS.
- Unify classifier and recovery taxonomies.
- Deploy the real-time Arducam dashboard.

### Figure / Table Suggestions

- `fig7_classifier_headline.png`
- `fig1_recovery_ranking.png`
- `fig2_score_composition.png`
- `fig3_value_vs_score.png`
- `fig8_pipeline.png`

## Draft 2 - Sustainability / Material-Recovery Focus

### Short Title Option

Seeing Value in Waste: AI-Guided Prioritization of Recoverable E-Waste Materials

### Research Question / Objective

Can AI help prioritize e-waste components that contain the most valuable and
recoverable materials?

### Introduction

- E-waste contains copper, gold, silver, palladium, lithium, cobalt, nickel, and rare-earth materials.
- Manual sorting is slow and can miss high-value components.
- AI can help identify which components should be recovered first.

### Methods

- Camera/image input is processed by a MobileNetV2 classifier.
- Predicted components are linked to a material database.
- Database fields include estimated value, copper percent, precious-metal percent, and priority.
- Recovery score combines economic and material factors.
- Dataset augmentation improved class balance and expanded the working dataset to 27,380 images.

### Results

- 90.7% validation accuracy.
- CPUs, connectors, laptop motherboards, smartphones, RAM, and GPUs ranked highest.
- Cables contain high copper content and remain important even when dollar value is lower.
- The material table translates component identity into recovery decisions.

### Discussion

- The project connects what the model sees to what recyclers may want to recover.
- The ranking can help prioritize effort in a mixed e-waste stream.
- Material composition values are literature-based estimates, not direct lab measurements.

### Conclusion

Computer vision can support e-waste recovery by identifying components and
ranking them according to recoverable material value.

### Future Work

- Validate material estimates experimentally.
- Add more labeled examples of high-value components.
- Train the YOLO detector for real-time bounding-box detection.
- Test the dashboard in a live sorting scenario.

### Figure / Table Suggestions

- `fig1_recovery_ranking.png`
- `fig9_material_table.png`
- `fig4_copper_content.png`
- `fig5_precious_metals.png`
- `fig8_pipeline.png`

## Draft 3 - Balanced / Conference-Ready

### Short Title Option

AI-Assisted E-Waste Recovery: Connecting Computer Vision to Material-Recovery Decisions

### Research Question / Objective

Can a computer-vision system identify e-waste components and prioritize them for
material recovery using an interpretable scoring model?

### Introduction

- E-waste locks away recoverable copper, gold, silver, palladium, and other critical materials.
- Sorting is a bottleneck for recovery.
- Classification alone identifies objects but does not provide recovery guidance.
- This project links vision predictions to material-recovery decisions.

### Methods

- MobileNetV2 classifier trained on 23,960 images across 18 classes.
- 80/20 train-validation split and 10 training epochs.
- Dataset balancing expanded the working dataset to 27,380 images.
- Component prediction is linked to a 27-component material database.
- Recovery score:

```text
score = 0.50 * value + 0.30 * copper + 0.20 * precious metals
```

### Results

- 90.7% validation accuracy.
- Recovery score calculated for 27 components.
- Highest-priority targets: CPU, connector, laptop motherboard, smartphone, RAM, GPU, PCB, cable, tablet, and SSD.
- Dataset balancing removed underrepresented canonical classes below the 900-image threshold.
- Recovery ranking shows how computer vision output can become decision support.

### Discussion

- Main contribution: an interpretable bridge from recognition to recovery prioritization.
- Transparent weights make the system explainable to reviewers and recyclers.
- Current limitations: custom YOLO detector is not fully trained, material values are literature-based, and classifier/recovery taxonomies still need unification.

### Conclusion

AI can both identify e-waste components and rank them for recovery, transforming
classification into actionable material-recovery support.

### Future Work

- Train and evaluate the custom YOLO detector.
- Report precision, recall, mAP, latency, and FPS.
- Deploy the Arducam dashboard.
- Validate material composition estimates using measured samples.

### Figure / Table Suggestions

- Must include: `fig7_classifier_headline.png`, `fig1_recovery_ranking.png`, `fig9_material_table.png`, `fig8_pipeline.png`.
- Strong support: `fig2_score_composition.png`, `fig3_value_vs_score.png`.
- Optional if space allows: `fig4_copper_content.png`, `fig5_precious_metals.png`, `fig6_dataset_distribution.png`.

## Recommendation

Use Draft 3. It is the strongest final-poster version because it reads like a
computer science research poster while still explaining the sustainability
impact. Draft 1 is best for a technical AI audience. Draft 2 is best for an
environmental or sustainability audience.

