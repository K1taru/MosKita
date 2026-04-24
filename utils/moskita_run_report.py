from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_duration(seconds: float) -> str:
    total_seconds = int(round(max(float(seconds), 0.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def load_results_dataframe(results_csv: str | Path) -> pd.DataFrame:
    csv_path = Path(results_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    dataframe.columns = dataframe.columns.str.strip()
    return dataframe


def summarize_training_results(dataframe: pd.DataFrame) -> dict[str, Any]:
    required_columns = {
        "epoch",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    }
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"results.csv is missing required columns: {missing_columns}")

    epoch_numbers = dataframe["epoch"].astype(int)
    display_epochs = epoch_numbers + 1 if int(epoch_numbers.min()) == 0 else epoch_numbers
    best_index = int(dataframe["metrics/mAP50(B)"].idxmax())
    best_row = dataframe.iloc[best_index]

    epoch_seconds = None
    epoch_timing_table = None
    total_training_seconds = None
    average_epoch_seconds = None
    best_epoch_seconds = None

    if "time" in dataframe.columns:
        epoch_seconds = dataframe["time"].diff().fillna(dataframe["time"]).astype(float)
        total_training_seconds = float(dataframe["time"].iloc[-1])
        average_epoch_seconds = float(epoch_seconds.mean())
        best_epoch_seconds = float(epoch_seconds.iloc[best_index])
        epoch_timing_table = pd.DataFrame(
            {
                "epoch": display_epochs,
                "seconds": epoch_seconds.round(2),
                "formatted": epoch_seconds.map(format_duration),
            }
        )

    best_map50 = float(best_row["metrics/mAP50(B)"])
    summary = {
        "total_epochs": int(len(dataframe)),
        "best_epoch_index": best_index,
        "best_epoch": int(display_epochs.iloc[best_index]),
        "best_map50": best_map50,
        "best_map50_95": float(best_row["metrics/mAP50-95(B)"]),
        "best_precision": float(best_row["metrics/precision(B)"]),
        "best_recall": float(best_row["metrics/recall(B)"]),
        "performance_level": performance_band(best_map50),
        "display_epochs": display_epochs.tolist(),
        "epoch_seconds": epoch_seconds.tolist() if epoch_seconds is not None else None,
        "epoch_timing_table": epoch_timing_table,
        "total_training_seconds": total_training_seconds,
        "average_epoch_seconds": average_epoch_seconds,
        "best_epoch_seconds": best_epoch_seconds,
        "best_row": best_row.to_dict(),
    }

    for column in ("train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss", "train/dfl_loss", "val/dfl_loss"):
        if column in dataframe.columns:
            summary[column] = float(best_row[column])

    return summary


def performance_band(map50: float) -> str:
    if map50 > 0.85:
        return "PUBLISHABLE"
    if map50 > 0.75:
        return "GOOD"
    if map50 > 0.60:
        return "ACCEPTABLE"
    return "BELOW TARGET"


def print_training_results_summary(summary: Mapping[str, Any]) -> None:
    print("TRAINING RESULTS SUMMARY")
    print("=" * 50)
    print(f"total epochs      : {summary['total_epochs']}")
    print(f"best epoch        : {summary['best_epoch']}")
    print(f"best mAP@50       : {summary['best_map50']:.4f}")
    print(f"best mAP@50-95    : {summary['best_map50_95']:.4f}")
    print(f"best precision    : {summary['best_precision']:.4f}")
    print(f"best recall       : {summary['best_recall']:.4f}")

    if summary.get("total_training_seconds") is not None:
        print(f"total training    : {format_duration(summary['total_training_seconds'])}")
        print(f"avg epoch time    : {format_duration(summary['average_epoch_seconds'])}")
        print(f"best epoch time   : {format_duration(summary['best_epoch_seconds'])}")

    for column in ("train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss", "train/dfl_loss", "val/dfl_loss"):
        if column in summary:
            print(f"{column:<18}: {summary[column]:.4f}")

    print(f"performance level : {summary['performance_level']}")

    timing_table = summary.get("epoch_timing_table")
    if timing_table is not None:
        print()
        print("EPOCH TIMING BREAKDOWN")
        print(timing_table.to_string(index=False))


def _metric_values(source: Any, key: str) -> np.ndarray | None:
    values = getattr(source, key, None)
    if values is None and isinstance(source, Mapping):
        values = source.get(key)
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def _scalar_value(source: Any, key: str) -> float | None:
    value = getattr(source, key, None)
    if value is None and isinstance(source, Mapping):
        value = source.get(key)
    if value is None:
        return None
    return float(value)


def _coerce_support_counts(
    support_counts: Mapping[str | int, int] | None,
    class_names: Sequence[str],
) -> dict[str, int]:
    if support_counts is None:
        return {class_name: 0 for class_name in class_names}

    normalized = {class_name: 0 for class_name in class_names}
    for index, class_name in enumerate(class_names):
        if class_name in support_counts:
            normalized[class_name] = int(support_counts[class_name])
        elif index in support_counts:
            normalized[class_name] = int(support_counts[index])
    return normalized


def _extract_confusion_matrix(confusion_matrix: Any, num_classes: int) -> np.ndarray | None:
    if confusion_matrix is None:
        return None

    matrix = getattr(confusion_matrix, "matrix", confusion_matrix)
    if matrix is None:
        return None

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        return None
    if array.shape == (num_classes + 1, num_classes + 1):
        array = array[:num_classes, :num_classes]
    elif array.shape != (num_classes, num_classes):
        return None
    return array


def _normalized_confusion(confusion_matrix: np.ndarray | None) -> np.ndarray | None:
    if confusion_matrix is None:
        return None

    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    return np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums > 0,
    )


def _top_confused_pairs(
    confusion_matrix: np.ndarray | None,
    class_names: Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    if confusion_matrix is None:
        return []

    working = confusion_matrix.copy().astype(float)
    np.fill_diagonal(working, 0.0)
    ranked_indices = np.argsort(working.ravel())[::-1]

    pairs = []
    for flat_index in ranked_indices:
        actual_index, predicted_index = np.unravel_index(flat_index, working.shape)
        count = float(working[actual_index, predicted_index])
        if count <= 0:
            break
        pairs.append(
            {
                "actual_index": int(actual_index),
                "predicted_index": int(predicted_index),
                "actual_class": class_names[actual_index],
                "predicted_class": class_names[predicted_index],
                "count": count,
            }
        )
        if len(pairs) >= limit:
            break
    return pairs


def build_detection_report(
    box_metrics: Any,
    class_names: Sequence[str],
    support_counts: Mapping[str | int, int] | None = None,
    confusion_matrix: Any = None,
    top_k: int = 10,
) -> dict[str, Any]:
    ap50_values = _metric_values(box_metrics, "ap50")
    ap5095_values = _metric_values(box_metrics, "ap")
    precision_values = _metric_values(box_metrics, "p")
    recall_values = _metric_values(box_metrics, "r")

    if ap50_values is None or ap5095_values is None or precision_values is None or recall_values is None:
        raise ValueError("box_metrics must expose ap50, ap, p, and r arrays")

    supports = _coerce_support_counts(support_counts, class_names)
    confusion_array = _extract_confusion_matrix(confusion_matrix, len(class_names))
    normalized_confusion = _normalized_confusion(confusion_array)

    per_class = []
    for index, class_name in enumerate(class_names):
        precision = float(precision_values[index])
        recall = float(recall_values[index])
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append(
            {
                "class_index": index,
                "class_name": class_name,
                "ap50": float(ap50_values[index]),
                "ap50_95": float(ap5095_values[index]),
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "support": supports[class_name],
            }
        )

    sorted_by_ap50 = sorted(per_class, key=lambda row: row["ap50"], reverse=True)
    report = {
        "overall": {
            "map50": _scalar_value(box_metrics, "map50"),
            "map50_95": _scalar_value(box_metrics, "map"),
            "mean_precision": float(np.mean(precision_values)),
            "mean_recall": float(np.mean(recall_values)),
            "mean_f1": float(np.mean([row["f1"] for row in per_class])),
        },
        "per_class": per_class,
        "best_classes": sorted_by_ap50[: min(3, len(sorted_by_ap50))],
        "weakest_classes": list(reversed(sorted_by_ap50[-min(3, len(sorted_by_ap50)) :])),
        "most_confused_pairs": _top_confused_pairs(confusion_array, class_names, limit=top_k),
        "confusion_matrix": confusion_array,
        "normalized_confusion_matrix": normalized_confusion,
    }
    return report


def print_detection_report_summary(report: Mapping[str, Any]) -> None:
    overall = report["overall"]
    print("DETECTION EVALUATION SUMMARY")
    print("=" * 60)
    if overall["map50"] is not None:
        print(f"overall mAP@50      : {overall['map50']:.4f}")
    if overall["map50_95"] is not None:
        print(f"overall mAP@50-95   : {overall['map50_95']:.4f}")
    print(f"mean precision      : {overall['mean_precision']:.4f}")
    print(f"mean recall         : {overall['mean_recall']:.4f}")
    print(f"mean F1             : {overall['mean_f1']:.4f}")
    print()

    print(f"{'class':<28} {'AP50':>7} {'AP50-95':>9} {'P':>8} {'R':>8} {'F1':>8} {'support':>8}")
    print("-" * 88)
    for row in report["per_class"]:
        print(
            f"{row['class_name']:<28} {row['ap50']:>7.4f} {row['ap50_95']:>9.4f} "
            f"{row['precision']:>8.4f} {row['recall']:>8.4f} {row['f1']:>8.4f} {row['support']:>8}"
        )

    print("-" * 88)
    best = report["best_classes"]
    weakest = report["weakest_classes"]
    print("best classes        : " + ", ".join(f"{row['class_name']} ({row['ap50']:.3f})" for row in best))
    print("weakest classes     : " + ", ".join(f"{row['class_name']} ({row['ap50']:.3f})" for row in weakest))

    confused_pairs = report["most_confused_pairs"]
    if confused_pairs:
        top_pair = confused_pairs[0]
        print(
            "most confused pair  : "
            f"{top_pair['actual_class']} -> {top_pair['predicted_class']} "
            f"({top_pair['count']:.0f})"
        )


def plot_detection_report(report: Mapping[str, Any]) -> tuple[Any, np.ndarray]:
    per_class = report["per_class"]
    class_names = [row["class_name"] for row in per_class]
    ap50_values = [row["ap50"] for row in per_class]
    precision_values = [row["precision"] for row in per_class]
    recall_values = [row["recall"] for row in per_class]
    f1_values = [row["f1"] for row in per_class]
    support_values = [row["support"] for row in per_class]
    confused_pairs = report["most_confused_pairs"]
    normalized_confusion = report["normalized_confusion_matrix"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = np.asarray(axes)

    threshold_lines = [
        (0.60, "#FFA500", "Acceptable"),
        (0.75, "#2196F3", "Good"),
        (0.85, "#4CAF50", "Publishable"),
    ]
    ap_colors = plt.cm.RdYlGn(np.clip(ap50_values, 0.0, 1.0))
    ap_bars = axes[0, 0].bar(class_names, ap50_values, color=ap_colors, edgecolor="black", linewidth=0.4)
    for bar, value in zip(ap_bars, ap50_values):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for threshold, color, label in threshold_lines:
        axes[0, 0].axhline(threshold, color=color, linestyle="--", linewidth=1.0, label=label)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Per-Class AP@50", fontweight="bold")
    axes[0, 0].set_ylabel("AP@50")
    axes[0, 0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0, 0].legend(fontsize=8)

    x = np.arange(len(class_names))
    width = 0.25
    axes[0, 1].bar(x - width, precision_values, width, label="Precision", color="#4C72B0")
    axes[0, 1].bar(x, recall_values, width, label="Recall", color="#DD8452")
    axes[0, 1].bar(x + width, f1_values, width, label="F1", color="#55A868")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Precision / Recall / F1", fontweight="bold")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend(fontsize=8)

    support_bars = axes[0, 2].bar(class_names, support_values, color="#C44E52", edgecolor="black", linewidth=0.4)
    for bar, value in zip(support_bars, support_values):
        axes[0, 2].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0, 2].set_title("Validation Support by Class", fontweight="bold")
    axes[0, 2].set_ylabel("Annotations")
    axes[0, 2].tick_params(axis="x", rotation=45, labelsize=8)

    axes[1, 0].scatter(support_values, ap50_values, s=80, c=ap50_values, cmap="RdYlGn", edgecolor="black", linewidth=0.5)
    for support, ap50, class_name in zip(support_values, ap50_values, class_names):
        axes[1, 0].annotate(class_name, (support, ap50), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[1, 0].set_title("AP@50 vs Validation Support", fontweight="bold")
    axes[1, 0].set_xlabel("Support")
    axes[1, 0].set_ylabel("AP@50")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(alpha=0.25)

    if confused_pairs:
        pair_labels = [f"{row['actual_class']} -> {row['predicted_class']}" for row in confused_pairs]
        pair_counts = [row["count"] for row in confused_pairs]
        axes[1, 1].barh(pair_labels, pair_counts, color="#8172B3")
        axes[1, 1].set_title("Top Confused Class Pairs", fontweight="bold")
        axes[1, 1].set_xlabel("Count")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.02,
            0.98,
            "No off-diagonal confusion counts were available.",
            va="top",
            ha="left",
            fontsize=11,
        )

    if normalized_confusion is not None:
        image = axes[1, 2].imshow(normalized_confusion, cmap="Blues", vmin=0.0, vmax=1.0)
        axes[1, 2].set_title("Normalized Confusion Matrix", fontweight="bold")
        axes[1, 2].set_xticks(np.arange(len(class_names)))
        axes[1, 2].set_yticks(np.arange(len(class_names)))
        axes[1, 2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        axes[1, 2].set_yticklabels(class_names, fontsize=7)
        axes[1, 2].set_xlabel("Predicted")
        axes[1, 2].set_ylabel("Actual")
        fig.colorbar(image, ax=axes[1, 2], fraction=0.046, pad=0.04)
    else:
        axes[1, 2].axis("off")
        best_classes = report["best_classes"]
        weakest_classes = report["weakest_classes"]
        summary_lines = ["Best AP@50 classes:"]
        summary_lines.extend(f"- {row['class_name']}: {row['ap50']:.3f}" for row in best_classes)
        summary_lines.append("")
        summary_lines.append("Weakest AP@50 classes:")
        summary_lines.extend(f"- {row['class_name']}: {row['ap50']:.3f}" for row in weakest_classes)
        axes[1, 2].text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=11)

    fig.suptitle("MosKita Detection Evaluation Overview", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig, axes