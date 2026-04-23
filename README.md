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

### Available Data (V1)

| Source | Type | Images | Annotations | Classes → V1 | License |
|---|---|---:|---:|---|---|
| **Adnans Breeding Place** | Outsource (Roboflow) | 4,425 | 4,895 | Bottle, Coconut-Exocarp → `uncovered_container`; Tire → `discarded_tire`; Drain-Inlet → `drain_inlet`; Vase → `flower_pot` | CC BY 4.0 |
| **Faiyaz MosquitoFusion** | Outsource (Roboflow) | 1,047 | 1,454 | Breeding Place → `uncovered_container` (Mosquito / Swarm removed) | CC BY 4.0 |
| **Roboflow Public** | Outsource (Roboflow) | 288 | 124 | bucket → `bucket`; puddle → `stagnant_puddle`; tire → `discarded_tire` | CC BY 4.0 |
| **Local — plastic_drum** | Self-collected (raw) | 107 | TBD | → `plastic_drum` | Own |
| **Local — bucket** | Self-collected (raw) | 84 + 42 | TBD | → `bucket` | Own |
| **Local — flower_pot** | Self-collected (raw) | 76 | TBD | → `flower_pot` | Own |
| **Local — styrofoam_container** | Self-collected (raw) | 55 | TBD | → `styrofoam_container` | Own |
| **Local — batch / multi_class** | Self-collected (raw) | 31 + 8 | TBD | mixed classes | Own |
| **Total** | | **~6,163** | **~6,473+** | — | — |

> Raw local images are resized to 1280×1280 via `utils/image_resizer.py` and await annotation in Roboflow.

### Class Coverage Status

| Class | Annotated | Gap |
|---|---|---|
| `discarded_tire` | ~1,212 (outsource) | ✅ Good |
| `flower_pot` | ~1,518 (outsource) | ✅ Good |
| `uncovered_container` | ~3,451 (outsource) | ✅ Strong |
| `drain_inlet` | ~230 (outsource) | ⚠️ Moderate — collect more |
| `stagnant_puddle` | ~56 (outsource) | ❌ Low — collect more |
| `plastic_drum` | 0 | ❌ Annotate 107 local raw imgs |
| `bucket` | ~7 (outsource) | ❌ Annotate 126 local raw imgs |
| `styrofoam_container` | 0 | ❌ Annotate 55 local raw imgs |

- **Annotation**: Roboflow (YOLOv8 format)
- **Assembly**: `training.ipynb` Section 3 — toggle sources and rebuild via `scripts/remap_yolo_dataset.py`
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
MosKita/
├── data/
│   ├── raw/                          # local photos, resized to 1280×1280 (moskita_*.jpg)
│   │   └── logs/                     # conversion_log.csv
│   ├── annotated/
│   │   ├── outsource/
│   │   │   ├── adnans/
│   │   │   │   └── Breeding Place Detection/  # 4,425 imgs — CC BY 4.0
│   │   │   ├── faiyazabdullah/
│   │   │   │   └── MosquitoFusion Dataset/    # 1,047 imgs — CC BY 4.0
│   │   │   └── roboflow/                      # 288 imgs  — CC BY 4.0
│   │   ├── train/ val/ test/          # assembled by training.ipynb §3
│   └── data.yaml
├── models/
│   ├── runs/                          # YOLOv8 training outputs
│   └── exports/                       # moskita.onnx, moskita.tflite
├── notebooks/
│   ├── training.ipynb                 # main training + assembly
│   └── evaluation.ipynb
├── deploy/
│   └── pi_inference.py
├── scripts/
│   ├── remap_yolo_dataset.py          # merge & remap outsource datasets
│   └── class_maps/                    # JSON maps + v1_target_names.txt
├── utils/
│   └── image_resizer.py               # resize raw photos to 1280×1280
├── assets/
│   └── sample_detections/
└── Docs/
    ├── MOSKITA_CONTEXT.md
    ├── dengue-dataset-guide.html
    └── temp/                          # working notes
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

## V1 Detection Classes (8 classes)

```
0: discarded_tire
1: flower_pot
2: uncovered_container
3: drain_inlet
4: stagnant_puddle
5: plastic_drum
6: bucket
7: styrofoam_container
```

> Any detected object is a potential breeding site. Water-state is not part of any class name.

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
