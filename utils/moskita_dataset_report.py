from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLITS = ("train", "val", "test")


def _coerce_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def load_target_class_names(target_names_file: str | Path) -> list[str]:
    target_path = Path(target_names_file).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target names file does not exist: {target_path}")

    raw_text = target_path.read_text(encoding="utf-8")
    names = []
    for raw_line in raw_text.splitlines():
        for raw_name in raw_line.split(","):
            name = raw_name.strip()
            if name:
                names.append(name)
    return names


def _iter_label_rows(label_path: Path) -> Iterable[tuple[int, float, float]]:
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                continue
            yield class_id, width, height


def summarize_yolo_detection_dataset(
    dataset_root: str | Path,
    class_names: Sequence[str],
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    root = _coerce_root(dataset_root)
    num_classes = len(class_names)

    class_totals: Counter[int] = Counter()
    split_class_counters: dict[str, Counter[int]] = {}
    split_stats: dict[str, dict[str, float | int]] = {}
    invalid_class_ids: Counter[int] = Counter()

    all_widths: list[float] = []
    all_heights: list[float] = []
    all_areas: list[float] = []
    all_ratios: list[float] = []
    annotations_per_image: list[int] = []

    for split in splits:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"

        image_files = []
        if image_dir.exists():
            image_files = sorted(
                path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
            )

        label_files = []
        if label_dir.exists():
            label_files = sorted(path for path in label_dir.iterdir() if path.suffix.lower() == ".txt")

        split_counter: Counter[int] = Counter()
        split_annotations = 0

        for label_path in label_files:
            rows = list(_iter_label_rows(label_path))
            annotations_per_image.append(len(rows))
            split_annotations += len(rows)

            for class_id, width, height in rows:
                if 0 <= class_id < num_classes:
                    split_counter[class_id] += 1
                    class_totals[class_id] += 1
                else:
                    invalid_class_ids[class_id] += 1

                all_widths.append(width)
                all_heights.append(height)
                all_areas.append(width * height)
                all_ratios.append(width / height if height > 0 else 0.0)

        split_class_counters[split] = split_counter
        split_stats[split] = {
            "images": len(image_files),
            "labels": len(label_files),
            "annotations": split_annotations,
            "avg_annotations_per_image": (
                split_annotations / len(label_files) if label_files else 0.0
            ),
        }

    total_annotations = int(sum(class_totals.values()))
    total_images = int(sum(int(stats["images"]) for stats in split_stats.values()))
    total_labels = int(sum(int(stats["labels"]) for stats in split_stats.values()))

    class_count_map = {class_names[index]: int(class_totals.get(index, 0)) for index in range(num_classes)}
    split_count_map = {
        split: {class_names[index]: int(counter.get(index, 0)) for index in range(num_classes)}
        for split, counter in split_class_counters.items()
    }

    max_class_count = max(class_count_map.values(), default=0)
    balance_rows = []
    for class_name in class_names:
        count = class_count_map[class_name]
        deficit = max_class_count - count
        share = (count / total_annotations * 100.0) if total_annotations else 0.0
        ratio = (count / max_class_count) if max_class_count else 0.0
        balance_rows.append(
            {
                "class_name": class_name,
                "annotations": count,
                "share_pct": share,
                "deficit_to_max": deficit,
                "class_ratio": ratio,
            }
        )

    small_count = sum(1 for area in all_areas if area < 0.0032)
    medium_count = sum(1 for area in all_areas if 0.0032 <= area < 0.04)
    large_count = sum(1 for area in all_areas if area >= 0.04)

    bbox_stats = {
        "count": len(all_areas),
        "median_area": float(np.median(all_areas)) if all_areas else 0.0,
        "mean_area": float(np.mean(all_areas)) if all_areas else 0.0,
        "mean_aspect_ratio": float(np.mean(all_ratios)) if all_ratios else 0.0,
        "median_aspect_ratio": float(np.median(all_ratios)) if all_ratios else 0.0,
        "mean_annotations_per_image": (
            float(np.mean(annotations_per_image)) if annotations_per_image else 0.0
        ),
        "max_annotations_per_image": max(annotations_per_image, default=0),
        "small_objects": small_count,
        "medium_objects": medium_count,
        "large_objects": large_count,
    }

    return {
        "dataset_root": root,
        "splits": tuple(splits),
        "class_names": list(class_names),
        "split_stats": split_stats,
        "class_counts": class_count_map,
        "split_class_counts": split_count_map,
        "balance_rows": balance_rows,
        "bbox_stats": bbox_stats,
        "total_images": total_images,
        "total_labels": total_labels,
        "total_annotations": total_annotations,
        "invalid_class_ids": dict(invalid_class_ids),
    }


def print_detection_dataset_summary(summary: dict[str, Any]) -> None:
    split_stats = summary["split_stats"]
    class_counts = summary["class_counts"]
    balance_rows = summary["balance_rows"]
    bbox_stats = summary["bbox_stats"]

    print("DATASET SUMMARY")
    print("=" * 72)
    for split in summary["splits"]:
        stats = split_stats[split]
        status = "OK" if stats["images"] and stats["images"] == stats["labels"] else "WARN"
        print(
            f"{split:<6} [{status}]  images={int(stats['images']):>5}  "
            f"labels={int(stats['labels']):>5}  annotations={int(stats['annotations']):>6}"
        )
    print("-" * 72)
    print(f"total images       : {summary['total_images']}")
    print(f"total labels       : {summary['total_labels']}")
    print(f"total annotations  : {summary['total_annotations']}")
    print()

    print(f"{'class':<28} {'annotations':>12} {'share %':>9} {'ratio':>8} {'deficit':>10}")
    print("-" * 72)
    for row in balance_rows:
        print(
            f"{row['class_name']:<28} {row['annotations']:>12} "
            f"{row['share_pct']:>8.2f}% {row['class_ratio']:>8.3f} {row['deficit_to_max']:>10}"
        )

    if summary["invalid_class_ids"]:
        print()
        print(f"invalid class ids  : {summary['invalid_class_ids']}")

    print()
    print("BOUNDING BOX SNAPSHOT")
    print("-" * 72)
    print(f"boxes parsed            : {bbox_stats['count']}")
    print(f"median area             : {bbox_stats['median_area']:.5f}")
    print(f"mean aspect ratio       : {bbox_stats['mean_aspect_ratio']:.3f}")
    print(f"mean annotations/image  : {bbox_stats['mean_annotations_per_image']:.2f}")
    print(
        "object size buckets      : "
        f"small={bbox_stats['small_objects']}, "
        f"medium={bbox_stats['medium_objects']}, "
        f"large={bbox_stats['large_objects']}"
    )


def plot_detection_dataset_overview(summary: dict[str, Any]) -> tuple[Any, np.ndarray]:
    class_names = summary["class_names"]
    splits = list(summary["splits"])
    split_stats = summary["split_stats"]
    class_counts = summary["class_counts"]
    split_class_counts = summary["split_class_counts"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    x = np.arange(len(splits))
    width = 0.24
    images = [split_stats[split]["images"] for split in splits]
    labels = [split_stats[split]["labels"] for split in splits]
    annotations = [split_stats[split]["annotations"] for split in splits]
    axes[0].bar(x - width, images, width, label="images", color="#4C72B0")
    axes[0].bar(x, labels, width, label="labels", color="#DD8452")
    axes[0].bar(x + width, annotations, width, label="annotations", color="#55A868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_title("Split Volume", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)

    total_counts = [class_counts[name] for name in class_names]
    total_colors = plt.cm.Set3(np.linspace(0, 1, max(len(class_names), 1)))
    total_bars = axes[1].bar(class_names, total_counts, color=total_colors, edgecolor="black", linewidth=0.4)
    for bar, count in zip(total_bars, total_counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_title("Per-Class Annotation Totals", fontweight="bold")
    axes[1].set_ylabel("Annotations")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)

    class_positions = np.arange(len(class_names))
    split_width = 0.8 / max(len(splits), 1)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for index, split in enumerate(splits):
        counts = [split_class_counts[split][name] for name in class_names]
        offset = (index - (len(splits) - 1) / 2) * split_width
        axes[2].bar(
            class_positions + offset,
            counts,
            split_width,
            label=split,
            color=palette[index % len(palette)],
            edgecolor="white",
            linewidth=0.4,
        )
    axes[2].set_xticks(class_positions)
    axes[2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axes[2].set_title("Per-Split Class Distribution", fontweight="bold")
    axes[2].set_ylabel("Annotations")
    axes[2].legend(fontsize=9)

    fig.suptitle("MosKita Dataset Overview", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig, axes
