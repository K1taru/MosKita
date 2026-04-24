from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.moskita_dataset_report import (
    load_target_class_names,
    plot_detection_dataset_overview,
    summarize_yolo_detection_dataset,
)
from utils.moskita_run_report import (
    build_detection_report,
    plot_detection_report,
    summarize_training_results,
)


class MoskitaReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temp_dir.name)
        self.class_names = ["bucket", "drum", "discarded_tire"]

        for split in ("train", "val", "test"):
            (self.dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
            (self.dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)

        self._write_image("train", "img_1.jpg")
        self._write_image("train", "img_2.jpg")
        self._write_image("val", "img_3.jpg")
        self._write_image("test", "img_4.jpg")

        self._write_label(
            "train",
            "img_1.txt",
            [
                "0 0.5 0.5 0.20 0.10",
                "1 0.5 0.5 0.40 0.20",
            ],
        )
        self._write_label(
            "train",
            "img_2.txt",
            [
                "2 0.5 0.5 0.10 0.10",
            ],
        )
        self._write_label(
            "val",
            "img_3.txt",
            [
                "0 0.5 0.5 0.30 0.20",
                "0 0.5 0.5 0.25 0.15",
                "2 0.5 0.5 0.20 0.25",
            ],
        )
        self._write_label(
            "test",
            "img_4.txt",
            [
                "1 0.5 0.5 0.30 0.30",
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_image(self, split: str, filename: str) -> None:
        (self.dataset_root / split / "images" / filename).write_bytes(b"placeholder")

    def _write_label(self, split: str, filename: str, rows: list[str]) -> None:
        (self.dataset_root / split / "labels" / filename).write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_dataset_summary_counts_split_and_class_statistics(self) -> None:
        summary = summarize_yolo_detection_dataset(self.dataset_root, self.class_names)

        self.assertEqual(summary["total_images"], 4)
        self.assertEqual(summary["total_labels"], 4)
        self.assertEqual(summary["total_annotations"], 7)
        self.assertEqual(summary["split_stats"]["train"]["images"], 2)
        self.assertEqual(summary["split_stats"]["val"]["annotations"], 3)
        self.assertEqual(
            summary["class_counts"],
            {"bucket": 3, "drum": 2, "discarded_tire": 2},
        )
        self.assertEqual(summary["split_class_counts"]["val"]["bucket"], 2)
        self.assertAlmostEqual(summary["bbox_stats"]["median_area"], 0.05, places=5)
        self.assertEqual(summary["bbox_stats"]["large_objects"], 4)

    def test_training_summary_handles_both_epoch_numbering_schemes(self) -> None:
        one_based = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "time": [15.0, 31.0, 48.0],
                "metrics/mAP50(B)": [0.45, 0.72, 0.61],
                "metrics/mAP50-95(B)": [0.20, 0.40, 0.35],
                "metrics/precision(B)": [0.50, 0.70, 0.64],
                "metrics/recall(B)": [0.40, 0.66, 0.58],
            }
        )
        zero_based = one_based.copy()
        zero_based["epoch"] = [0, 1, 2]

        one_based_summary = summarize_training_results(one_based)
        zero_based_summary = summarize_training_results(zero_based)

        self.assertEqual(one_based_summary["best_epoch"], 2)
        self.assertEqual(zero_based_summary["best_epoch"], 2)
        self.assertEqual(one_based_summary["performance_level"], "ACCEPTABLE")
        self.assertAlmostEqual(one_based_summary["average_epoch_seconds"], 16.0)
        self.assertEqual(list(one_based_summary["epoch_timing_table"]["epoch"]), [1, 2, 3])
        self.assertEqual(list(zero_based_summary["epoch_timing_table"]["epoch"]), [1, 2, 3])

    def test_target_name_loader_supports_csv_and_line_formats(self) -> None:
        csv_file = self.dataset_root / "target_csv.txt"
        csv_file.write_text("bucket, drum, discarded_tire\n", encoding="utf-8")
        self.assertEqual(
            load_target_class_names(csv_file),
            ["bucket", "drum", "discarded_tire"],
        )

        lines_file = self.dataset_root / "target_lines.txt"
        lines_file.write_text("bucket\n\ndrum\ndiscarded_tire\n", encoding="utf-8")
        self.assertEqual(
            load_target_class_names(lines_file),
            ["bucket", "drum", "discarded_tire"],
        )

    def test_detection_report_computes_f1_and_confusions(self) -> None:
        box_metrics = {
            "ap50": [0.91, 0.66, 0.51],
            "ap": [0.73, 0.42, 0.30],
            "p": [0.94, 0.71, 0.55],
            "r": [0.88, 0.63, 0.42],
            "map50": 0.6933,
            "map": 0.4833,
        }
        confusion_matrix = np.array(
            [
                [18, 2, 0, 0],
                [3, 14, 1, 0],
                [0, 2, 11, 0],
                [0, 0, 0, 0],
            ]
        )
        support_counts = {"bucket": 20, "drum": 18, "discarded_tire": 13}

        report = build_detection_report(
            box_metrics,
            self.class_names,
            support_counts=support_counts,
            confusion_matrix=confusion_matrix,
            top_k=3,
        )

        self.assertEqual(report["best_classes"][0]["class_name"], "bucket")
        self.assertEqual(report["weakest_classes"][0]["class_name"], "discarded_tire")
        self.assertEqual(report["most_confused_pairs"][0]["actual_class"], "drum")
        self.assertEqual(report["most_confused_pairs"][0]["predicted_class"], "bucket")
        self.assertAlmostEqual(report["per_class"][1]["f1"], 2 * 0.71 * 0.63 / (0.71 + 0.63))
        self.assertEqual(report["confusion_matrix"].shape, (3, 3))
        self.assertEqual(report["normalized_confusion_matrix"].shape, (3, 3))

    def test_plot_helpers_render_expected_layouts(self) -> None:
        dataset_summary = summarize_yolo_detection_dataset(self.dataset_root, self.class_names)
        dataset_fig, dataset_axes = plot_detection_dataset_overview(dataset_summary)

        box_metrics = {
            "ap50": [0.91, 0.66, 0.51],
            "ap": [0.73, 0.42, 0.30],
            "p": [0.94, 0.71, 0.55],
            "r": [0.88, 0.63, 0.42],
            "map50": 0.6933,
            "map": 0.4833,
        }
        report = build_detection_report(
            box_metrics,
            self.class_names,
            support_counts=dataset_summary["split_class_counts"]["val"],
            confusion_matrix=np.eye(4),
        )
        detection_fig, detection_axes = plot_detection_report(report)

        self.assertEqual(len(dataset_axes), 3)
        self.assertEqual(detection_axes.shape, (2, 3))

        plt.close(dataset_fig)
        plt.close(detection_fig)


if __name__ == "__main__":
    unittest.main()