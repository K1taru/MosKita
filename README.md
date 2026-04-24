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
| **Roboflow Public** | Outsource (Roboflow) | 288 | 409 | bucket → `bucket`; puddle → `stagnant_puddle`; tire → `discarded_tire` | CC BY 4.0 |
| **K1taru Self-Curated Export** | Local annotated (Roboflow) | 1,245 | 1,725 | basin → `uncovered_container`; bucket → `bucket`; drum → `drum`; plant pot → `flower_pot`; styrofoam container → `styrofoam_container` | Private |
| **Local Raw Archive** | Self-collected (organized raw) | 403 | — | source archive used to build `data/annotated/k1taru` | Own |
| **Train-ready total** | | **7,005** | **8,483** | — | — |

> `data/annotated/k1taru/` is the train-ready local source. The 403 images under `data/raw/` are the organized source archive and are not counted again in the train-ready total.

### Class Coverage Status

| Class | Annotated | Gap |
|---|---|---|
| `discarded_tire` | ~2,018 (outsource only) | ✅ Strong |
| `flower_pot` | ~3,068 (outsource + k1taru plant pot) | ✅ Strong |
| `uncovered_container` | ~6,148 (outsource + k1taru basin remap) | ✅ Strong |
| `drain_inlet` | ~1,353 (outsource only) | ✅ Good |
| `stagnant_puddle` | ~145 (outsource only) | ⚠️ Low — collect more |
| `drum` | 388 (k1taru drum) | ✅ Ready |
| `bucket` | ~581 (roboflow + k1taru bucket) | ✅ Ready |
| `styrofoam_container` | 185 (k1taru) | ⚠️ Moderate |

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
│   │   ├── k1taru/                        # 1,245 imgs — private self-curated Roboflow export
│   │   ├── outsource/
│   │   │   ├── adnans/
│   │   │   │   └── Breeding Place Detection/  # 4,425 imgs — CC BY 4.0
│   │   │   ├── faiyazabdullah/
│   │   │   │   └── MosquitoFusion Dataset/    # 1,047 imgs — CC BY 4.0
│   │   │   └── roboflow/                      # 288 imgs  — CC BY 4.0
│   │   ├── train/ val/ test/               # assembled by training.ipynb §3
│   │   └── data.yaml                       # assembled V1 dataset config
├── models/
│   ├── runs/                          # YOLOv8 training outputs
│   └── exports/                       # moskita.onnx, moskita.tflite
├── notebooks/
│   ├── training.ipynb                 # main training + assembly
│   └── evaluation.ipynb
├── deploy/
│   └── pi_inference.py
├── MosKita-Inference/                 # React browser inference dashboard
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
    data='data/annotated/data.yaml',
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

### Web Inference Dashboard (React)
```bash
cd MosKita-Inference
npm install

# Optional: place your exported ONNX model here for auto-load
cp ../models/exports/moskita.onnx public/models/moskita.onnx

# Starts a responsive browser client with camera + uploaded-video inference
npm run dev -- --host
```

- Default model path: `MosKita-Inference/public/models/moskita.onnx`
- If you do not copy the model into `public/models/`, the UI also lets you upload `moskita.onnx` directly at runtime.
- Camera access works only in a secure context: use `http://localhost:5173` on the same machine, or serve the dashboard over HTTPS when opening it from another device (for example, via a LAN IP on a phone).
- The dashboard shows current FPS, average FPS, last latency, average latency, p95 latency, frame count, and latest detections.

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
5: drum
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
