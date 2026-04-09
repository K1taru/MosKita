# MosKita — Project Context for Claude Code

> **Name origin:** "MosKita" — from *"makita"* (Tagalog: to see/to be seen) + mosquito. Stylized as **Mos**Kita.
> **Tagline:** *"makita ko kita"* — I can see you (addressed to the mosquito breeding site).

---

## 1. Project Summary

**MosKita** is a YOLOv8-based object detection system that automatically identifies dengue mosquito (*Aedes aegypti* / *Aedes albopictus*) breeding sites from photographs or a live camera feed. The model is trained on a **self-curated image dataset** collected from Metro Manila barangays, vacant lots, cemeteries, and construction sites.

The end goal is **edge deployment on a Raspberry Pi 5** with a camera module for real-time barangay-level surveillance — a low-cost, autonomous tool for local government units (LGUs) and barangay health workers.

---

## 2. Developer Info

| Field | Value |
|---|---|
| **Name** | GitHub: K1taru |
| **Course** | |
| **School** | |
| **Subject** | |
| **Section** | |

---

## 3. Hardware

### Development Machine
| Component | Spec |
|---|---|
| **Laptop** | Lenovo Legion 5 |
| **CPU** | AMD Ryzen 7 4800H |
| **GPU** | NVIDIA RTX 2060 6GB GDDR6 |
| **RAM** | 16GB DDR4 @ 3200MHz |
| **OS** | Dual boot: Windows 11 + Linux Mint |

### Deployment Device
| Component | Spec |
|---|---|
| **Device** | Raspberry Pi 5 |
| **RAM** | 8GB |
| **Hostname** | `raspy@k1taru` |
| **Role** | Edge inference (ONNX / TFLite) + camera module |


---

## 4. Model Architecture

| Decision | Choice | Reason |
|---|---|---|
| **Framework** | YOLOv8s (small) | Best accuracy-to-VRAM ratio for RTX 2060 6GB |
| **Task** | Object Detection | Bounding box + class label + confidence — most useful for deployment |
| **Export** | ONNX / TFLite | For Pi 5 edge inference |
| **Batch size** | 16 | Fits within 6GB VRAM |
| **Epochs** | 50–100 | Standard for fine-tuning on custom dataset |

> **Why not classification?** A classifier only says "breeding site present" — object detection localizes it, which is far more actionable for field workers.
> **Why not segmentation?** Overkill for Phase 1. Bounding boxes are sufficient for detection and deployment.

---

## 5. Detection Classes

Two-part label schema: `{container_type}_{water_status}`

### Priority Classes (Phase 1 — start here)
```
plastic_drum_open
plastic_drum_covered
metal_drum_open
discarded_tire_pooled
discarded_tire_dry
flower_pot_saucer_wet
flower_pot_saucer_dry
tarpaulin_pooled
uncovered_water_container_wet
uncovered_water_container_dry
```

### Extended Classes (Phase 2)
```
bamboo_stump_open
coconut_shell_pooled
construction_block_pooled
clogged_gutter
tin_can_pooled
cemetery_vase_wet
cemetery_vase_dry
```

---

## 6. Dataset Specification

### Volume Targets
| Phase | Images/Class | Classes | Total Images | Goal |
|---|---|---|---|---|
| Phase 1 (PoC) | 80–150 | 5 | ~500 | Validate concept |
| Phase 2 (Demo) | 200–300 | 10 | ~2,500 | Thesis/paper submission |
| Phase 3 (Deploy) | 500+ | 10–15 | ~6,000+ | Real-world use |

### Dataset Split
```
70% Training
20% Validation
10% Test   ← NEVER augment, NEVER touch until final eval
```

### Shot Protocol Per Object
| Variation Axis | Options | Coverage |
|---|---|---|
| Distance | Close (1–1.5m), Medium (2–4m), Far (5–10m) | 3 |
| Angle | Eye-level (0°), Diagonal (45°), Top-down (90°) | 3 |
| Lighting | Overcast, Bright sun, Shade | 2 |
| Water state | Wet (water visible), Dry | 2 |
| Context | Isolated, Cluttered scene | 2 |

**Formula:** 3×3×2×2×2 = **~72 raw shots per class** minimum
**After Roboflow augmentation:** ~600 effective samples per class

### Shot Distribution Per Class (per 100 images)
- Close (1–1.5m): 20 shots — detail, water state, material
- Medium (2–4m): 50 shots — **primary generalization distance**
- Far/scene (5–10m): 30 shots — multi-object, real deployment context

### Augmentations (via Roboflow)
- Horizontal flip
- 90° rotation
- Brightness ±30%
- Slight blur (phone camera simulation)
- Mosaic (multi-object scenes)
- Random crop (partial visibility)

---

## 7. Annotation

### Tool
**Roboflow** (free tier) — annotation + augmentation + YOLOv8 format export

### Bounding Box Rules
```
DO:
  - Box tightly around the container/object only
  - Include water surface if visible
  - One box per object instance
  - Annotate ALL visible instances in the frame

DON'T:
  - Box the entire scene or background
  - Skip partially visible objects
  - Include shadow/ground in the box
  - Leave unannotated instances (YOLO treats them as background = false negatives)
```

---

## 8. Breeding Site Reference

Complete list of objects/contexts classified as dengue breeding sites:

### Category A — WHO Recognized / High Priority
- Plastic drums (open) — #1 for *Ae. aegypti* in PH studies (40.2% of pupae)
- Metal drums / galvanized barrels (29.6%)
- Discarded rubber tires — key for *Ae. albopictus*
- Earthen/clay pots (palayok, burnay)
- Concrete/cement tanks (tinaja)
- Overhead and underground water tanks (uncovered)
- Buckets (left outdoors)
- Jerry cans / water jugs (open)
- Flower vases + pot saucers
- Water storage jars
- Tin cans (discarded)
- Cisterns (above/below ground)

### Category B — Household / Indoor
- Refrigerator drip trays
- AC drip pans / drain lines
- Pet water bowls (unchanged)
- Animal drinking troughs
- Unused bathtubs / bathroom fixtures
- Washing machine trays
- Mop buckets left outdoors
- Unused cooking pots outdoors

### Category C — Natural / Vegetation
- Bamboo stumps — #1 natural site for *Ae. albopictus* (28.5%)
- Coconut shells
- Tree holes / trunk cavities
- Leaf axils (banana, pandan, gabi/taro, bromeliad)
- Fallen leaves forming cups
- Cut bamboo internodes
- Coconut husk / areca shells

### Category D — Construction / Infrastructure
- Hollow CHB blocks stacked outdoors
- PVC / metal pipes stored horizontally
- Tarps / plastic sheets with rainwater pools
- Roof gutters / eaves troughs (clogged)
- Clogged downspouts
- Septic tank lids (poorly sealed)
- Manholes with water
- Foundation excavation pools

### Category E — Vehicle / Transport
- Discarded tires (any size)
- Tire rims with pooling water
- Unused vehicle bodies / jeepney recesses
- Boat bilges / unused bangka
- Hubcaps (discarded, outdoors)

### Category F — Discarded / Waste
- Discarded Styrofoam boxes (seafood delivery)
- Pooled plastic bags
- Discarded takeout containers / lids
- Broken toilets / sinks
- Old footwear holding water
- Discarded tin roofing sheets

### Category G — Cemetery-Specific
- Flower vases on grave markers
- Water jars / ceramic offering containers
- Candle holders with rainwater
- Sunken grave slabs

### Category H — Environmental / Drainage
- Clogged street drains / estero edges
- Roadside puddles in potholes (>7 days)
- Flooded vacant lot depressions
- Neglected ornamental ponds / fish ponds
- Abandoned swimming pools
- Turned-off water features / fountains

---

## 9. Breeding Site Risk Conditions

A container qualifies as HIGH RISK when it has:
1. **Stagnant water** (not flowing)
2. **Clean to slightly turbid** (not highly polluted — *Ae. aegypti* avoids heavily polluted water)
3. **Water held for 7+ days** (egg-to-adult cycle requires ~7–10 days)
4. **Shaded / semi-shaded location** — mosquitoes strongly prefer shade
5. **Container opening is uncovered or has gaps**

Additional risk multipliers:
- Rainwater (more productive than tap water alone)
- Algae presence (indicates stagnation)
- Organic debris floating in water
- Located in dense vegetation

---

## 10. Deployment Architecture

```
Training Pipeline (Legion 5):
  Raw Images → Roboflow (annotate + augment) → YOLOv8s fine-tune (RTX 2060) → Evaluate mAP

Deployment Pipeline (Pi 5):
  YOLOv8s weights → ONNX / TFLite export → Pi 5 inference (pi_inference.py) → Camera module
```

### Target Metrics
| Metric | Acceptable | Good | Publishable |
|---|---|---|---|
| mAP@50 | >0.60 | >0.75 | >0.85 |
| Precision | >0.65 | >0.78 | >0.88 |
| Recall | >0.60 | >0.75 | >0.83 |
| Inference (Pi 5) | <500ms | <200ms | <100ms |

---

## 11. Recommended Project Structure

```
moskita/
│
├── data/
│   ├── raw/                        # original unedited photos
│   │   ├── plastic_drum_open/
│   │   ├── discarded_tire_pooled/
│   │   └── ...
│   ├── annotated/                  # Roboflow YOLOv8 export
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   ├── data.yaml                   # YOLOv8 dataset config
│   └── collection_log.csv          # date, collector, location, class, count
│
├── models/
│   ├── runs/                       # YOLO training outputs (weights, plots)
│   └── exports/
│       ├── moskita.onnx            # for Pi 5 deployment
│       └── moskita.tflite
│
├── notebooks/
│   ├── 01_eda.ipynb                # dataset exploration
│   ├── 02_training.ipynb           # training runs
│   └── 03_evaluation.ipynb        # mAP, confusion matrix, per-class analysis
│
├── deploy/
│   ├── pi_inference.py             # runs on Raspberry Pi 5
│   └── requirements_pi.txt         # lightweight deps for Pi
│
├── scripts/
│   ├── split_dataset.py            # 70/20/10 split
│   └── check_annotations.py        # validate all labels present
│
├── assets/
│   └── sample_detections/          # output images with bounding boxes
│
├── MOSKITA_CONTEXT.md              # ← this file
└── README.md
```

---

## 12. data.yaml Template

```yaml
# MosKita — YOLOv8 Dataset Config
path: ./data/annotated
train: train/images
val: val/images
test: test/images

nc: 10  # number of classes (Phase 1)
names:
  0: plastic_drum_open
  1: plastic_drum_covered
  2: metal_drum_open
  3: discarded_tire_pooled
  4: discarded_tire_dry
  5: flower_pot_saucer_wet
  6: flower_pot_saucer_dry
  7: tarpaulin_pooled
  8: uncovered_container_wet
  9: uncovered_container_dry
```

---

## 13. Training Script (Baseline)

```python
# train.py — MosKita YOLOv8 Training
# Hardware: RTX 2060 6GB | Batch 16 | YOLOv8s

from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # load pretrained YOLOv8s

results = model.train(
    data='data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,          # fits RTX 2060 6GB
    name='moskita_v1',
    project='models/runs',
    patience=15,       # early stopping
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    augment=True,
    cache=False,       # set True if RAM allows
    device=0,          # GPU 0 (RTX 2060)
    verbose=True,
)
```

---

## 14. Pi 5 Inference Script (Baseline)

```python
# deploy/pi_inference.py — MosKita Edge Inference
# Hardware: Raspberry Pi 5 8GB + Camera Module

import cv2
from ultralytics import YOLO

model = YOLO('models/exports/moskita.onnx', task='detect')

CLASS_NAMES = [
    'plastic_drum_open', 'plastic_drum_covered', 'metal_drum_open',
    'discarded_tire_pooled', 'discarded_tire_dry',
    'flower_pot_saucer_wet', 'flower_pot_saucer_dry',
    'tarpaulin_pooled', 'uncovered_container_wet', 'uncovered_container_dry',
]

HIGH_RISK = {
    'plastic_drum_open', 'metal_drum_open', 'discarded_tire_pooled',
    'flower_pot_saucer_wet', 'tarpaulin_pooled', 'uncovered_container_wet'
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, iou=0.45)

    for r in results:
        for box in r.boxes:
            cls = CLASS_NAMES[int(box.cls)]
            conf = float(box.conf)
            risk = "HIGH RISK" if cls in HIGH_RISK else "MONITOR"
            color = (0, 0, 255) if risk == "HIGH RISK" else (0, 165, 255)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls} [{risk}] {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imshow('MosKita — Live Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 15. Development Timeline

| Week | Goal |
|---|---|
| Week 1 | Field collection sprint — 5 classes, 150 raw images/class |
| Week 2 | Annotation in Roboflow, augmentation, export YOLOv8 format |
| Week 3 | First training run on Legion 5 — baseline mAP evaluation |
| Week 4 | Active learning loop — fix weak classes, re-train |
| Week 5+ | Pi 5 deployment test, ONNX export, live inference demo |

---

## 16. Active Learning Loop

```
1. Train on current dataset
2. Run inference on unlabeled field photos
3. Find low-confidence detections (conf < 0.5)
4. Manually annotate those hard cases
5. Add to training set → re-train
6. Repeat until mAP@50 > 0.75
```

---

## 17. Key References

- PH Study: Cebu City pupal survey — plastic drums (#1 *Ae. aegypti*), bamboo stumps (#1 *Ae. albopictus*)
- WHO: *Ae. aegypti* breeds in clean-to-slightly-turbid water in artificial containers
- TIP Manila context: NCR has consistently high dengue incidence; Metro Manila barangays are primary deployment environment
- DOST-PCAARRD and DOH fund dengue vector control tools — potential grant/thesis pathway

---

## 18. Conversation Context

This project emerged from a brainstorming session on self-curated ML datasets. The full decision trail:

1. Started with general ML project ideas with self-curated data
2. Filtered for "niche" then "worth it to solve" — dengue breeding site detector ranked highest on impact + fundability + feasibility
3. Expanded the full taxonomy of breeding sites across 8 categories (WHO-recognized, household, natural, construction, vehicle, waste, cemetery, environmental)
4. Decided on **object detection (YOLOv8)** over classification for actionable bounding box output
5. Defined annotation schema: `{container_type}_{water_status}`
6. Shot protocol designed: 5-shot minimum per object, 3 distances, 3 angles, 2 lighting, wet/dry states
7. Phase 1 target: 5 classes, 500 images, RTX 2060 training, Pi 5 deployment
8. Originally named "DenguEye" → **renamed to MosKita** (from Tagalog *makita* = to see, + mosquito wordplay: *"makita ko kita"*)
9. Formal proposal document created for COE 005A Midterm Practical Examination
10. Dataset field guide created as a screenshot-ready wallpaper

---

*Generated from planning session — use this file as full context when continuing work in Claude Code.*
