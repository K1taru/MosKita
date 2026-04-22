# MosKita

**Automated dengue mosquito breeding site detection via YOLOv8 object detection.**

*"makita ko kita"* — I can see you (addressed to the mosquito breeding site).

---

## Overview

MosKita identifies *Aedes aegypti* and *Aedes albopictus* breeding containers from photographs and live camera feeds using a fine-tuned YOLOv8s model. Trained on a self-curated dataset from Metro Manila barangays, it targets **real-time edge deployment on Raspberry Pi 5** for autonomous, low-cost vector surveillance.

### Key Features
- **YOLOv8s object detection** — tight bounding boxes around breeding containers with confidence scores
- **8-class detection schema** — plastic drum, tire, flower pot, bucket, uncovered container, drain inlet, stagnant puddle, styrofoam container
- **ONNX/TFLite export** — optimized for Pi 5 inference (<500ms/frame)
- **Breeding site identification** — any detected object is flagged as a potential breeding site by definition
- **Field-ready taxonomy** — 8 categories of WHO-recognized breeding sites (household, natural, construction, cemetery, etc.)

---

## Hardware

| Role | Spec |
|---|---|
| **Training** | Lenovo Legion 5: RTX 2060 6GB, Ryzen 7 4800H |
| **Deployment** | Raspberry Pi 5: 8GB RAM + CSI camera module |

---

## Dataset & Training

- **Phase 1 target**: 500 images across 5–10 classes (80–150 per class)
- **Annotation**: Roboflow (YOLOv8 format)
- **Augmentation**: Horizontal flip, rotation, brightness, blur, mosaic
- **Split**: 70% train / 20% val / 10% test
- **Epochs**: 50–100 (early stopping at patience=15)

### Shot Protocol Per Class
- **Distances**: Close (1–1.5m), medium (2–4m), far (5–10m)
- **Angles**: Eye-level, diagonal (45°), top-down
- **Lighting**: Overcast, bright sun, shade
- **Context**: Isolated, cluttered scenes

> Shoot the object in whatever state you find it — wet, dry, empty, full. Detection = breeding site. Water state is not annotated.

---

## Project Structure

```
moskita/
├── data/
│   ├── raw/                    # unedited field photos
│   └── annotated/              # YOLOv8 train/val/test splits
├── models/
│   ├── runs/                   # training checkpoints & plots
│   └── exports/                # moskita.onnx, moskita.tflite
├── notebooks/
│   ├── eda.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
├── deploy/
│   ├── pi_inference.py         # Pi 5 live inference
│   └── requirements_pi.txt
├── scripts/
│   ├── split_dataset.py
│   └── check_annotations.py
└── Docs/
    └── MOSKITA_CONTEXT.md      # full project spec
```

---

## Quick Start

### Training (Legion 5)
```bash
pip install ultralytics opencv-python roboflow

python -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.train(
    data='data/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    patience=15,
    device=0,
)
"
```

### Deployment (Pi 5)
```bash
# On Raspberry Pi 5
pip install -r deploy/requirements_pi.txt
python deploy/pi_inference.py
```

---

## Metrics & Targets

| Metric | Acceptable | Good | Publishable |
|---|---|---|---|
| mAP@50 | >0.60 | >0.75 | >0.85 |
| Precision | >0.65 | >0.78 | >0.88 |
| Recall | >0.60 | >0.75 | >0.83 |
| Inference (Pi 5) | <500ms | <200ms | <100ms |

---

## Detection Classes (Phase 1)

```
plastic_drum_open, plastic_drum_covered, metal_drum_open,
discarded_tire_pooled, discarded_tire_dry,
flower_pot_saucer_wet, flower_pot_saucer_dry,
tarpaulin_pooled,
uncovered_container_wet, uncovered_container_dry
```

---

## References

- See [MOSKITA_CONTEXT.md](Docs/MOSKITA_CONTEXT.md) for full documentation
- WHO: *Aedes aegypti* breeds in clean-to-slightly-turbid water in artificial containers
- PH studies: CBC plastic drums (#1 for *Ae. aegypti*), bamboo stumps (#1 for *Ae. albopictus*)

---

## Developer

**GitHub:** K1taru

---

*For detailed specification, active learning loop, and full taxonomy, see [Docs/MOSKITA_CONTEXT.md](Docs/MOSKITA_CONTEXT.md)*
