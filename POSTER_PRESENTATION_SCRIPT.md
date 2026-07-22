# Poster Presentation Script

## 30-Second Version

This project uses computer vision to support e-waste recovery. The system classifies electronic waste components, looks up their estimated material composition, and ranks them using a recovery score based on economic value, copper content, and precious-metal content. The classifier reached 90.7% validation accuracy, and the recovery system ranks 27 components, with CPUs, connectors, laptop motherboards, smartphones, RAM, and GPUs as the highest-priority recovery targets.

## 2-Minute Walkthrough

My project is called "AI-Assisted E-Waste Recovery: Connecting Computer Vision to Material-Recovery Decisions."

The problem is that e-waste contains valuable materials, including copper, gold, silver, palladium, lithium, cobalt, and rare-earth materials, but sorting and prioritizing components is still difficult. A normal classifier can say what object is in an image, but it does not tell us which object should be recovered first.

So I built a pipeline that connects computer vision to material recovery. The system takes an image or camera frame, predicts the component class, looks that component up in a material database, and calculates a recovery score.

The recovery score is transparent:

```text
score = 0.50 * value + 0.30 * copper + 0.20 * precious metals
```

That means the score is based on estimated value per kilogram, copper content, and precious-metal content.

For the vision model, I used MobileNetV2 transfer learning. The classifier was trained on 23,960 labeled e-waste images across 18 classes, using an 80/20 train-validation split. It reached 90.7% validation accuracy.

I also performed dataset analysis and class-balancing augmentation. The expanded augmented dataset contains 27,380 images, and all canonical classes are above the 900-image low-frequency threshold.

The most important result is the recovery ranking. Across 27 components, the top targets were CPUs, connectors, laptop motherboards, smartphones, RAM, GPUs, PCBs, cables, tablets, and SSDs. This shows that the system is not only recognizing objects. It is translating recognition into recovery priority.

The current limitation is that the custom YOLO detector is still in progress. The live camera pipeline and dashboard exist, but the final YOLO `best.pt` model still needs bounding-box annotations and training. Also, the material composition values are literature-based estimates, not direct lab measurements.

The next step is to finish bounding-box labeling, train YOLO, evaluate precision, recall, mAP, and latency, and then deploy the Arducam real-time dashboard as the final demonstration.

## If A Professor Asks "What Is The Main Contribution?"

The main contribution is the bridge between vision and recovery decision-making. Instead of stopping at "this is a CPU" or "this is a PCB," the system estimates which detected component has the highest recovery priority based on value, copper, and precious metals.

## If A Professor Asks "Is YOLO Finished?"

Not fully yet. The YOLO infrastructure and live camera dashboard are in place, but the final custom YOLO detector still requires bounding-box annotations and training. Right now, the strongest completed model result is the MobileNetV2 classifier with 90.7% validation accuracy.

## If A Professor Asks "Are The Material Values Measured?"

No. The material composition estimates currently rely on published literature values and engineering estimates. Future work should validate them using direct measurement or lab analysis.

