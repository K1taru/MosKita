from __future__ import annotations

import time
from html import escape
from typing import Any, Mapping

try:
    from IPython.display import HTML, display
except Exception:  # pragma: no cover - notebook-only fallback
    HTML = None
    display = None

from utils.moskita_run_report import format_duration


class NotebookLiveTrainingProgress:
    """Render compact, live training progress inside a notebook output area."""

    def __init__(
        self,
        *,
        run_name: str,
        model_variant: str,
        refresh_seconds: float = 1.0,
        every_n_batches: int = 1,
        recent_epoch_count: int = 5,
    ) -> None:
        self.run_name = str(run_name)
        self.model_variant = str(model_variant)
        self.refresh_seconds = max(float(refresh_seconds), 0.25)
        self.every_n_batches = max(int(every_n_batches), 1)
        self.recent_epoch_count = max(int(recent_epoch_count), 1)

        self.started_at = 0.0
        self.epoch_started_at = 0.0
        self.last_render_at = 0.0
        self.current_epoch = 0
        self.total_epochs = 0
        self.batch_index = 0
        self.total_batches = 0
        self.epoch_rows: list[dict[str, float | int | None]] = []
        self.display_handle = None

    def install(self, model: Any) -> "NotebookLiveTrainingProgress":
        model.add_callback("on_train_start", self.on_train_start)
        model.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        model.add_callback("on_train_batch_end", self.on_train_batch_end)
        model.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)
        model.add_callback("on_train_end", self.on_train_end)
        return self

    def on_train_start(self, trainer: Any) -> None:
        now = time.perf_counter()
        self.started_at = now
        self.epoch_started_at = now
        self.last_render_at = 0.0
        self.total_epochs = int(getattr(trainer, "epochs", 0) or 0)
        self.total_batches = self._safe_total_batches(trainer)
        self.current_epoch = int(getattr(trainer, "start_epoch", 0) or 0) + 1
        self.batch_index = 0
        self.epoch_rows = []
        self._render(trainer, stage="starting", force=True)

    def on_train_epoch_start(self, trainer: Any) -> None:
        self.current_epoch = int(getattr(trainer, "epoch", 0) or 0) + 1
        self.total_batches = self._safe_total_batches(trainer)
        self.batch_index = 0
        self.epoch_started_at = time.perf_counter()
        self._render(trainer, stage="epoch_start", force=True)

    def on_train_batch_end(self, trainer: Any) -> None:
        self.batch_index += 1
        self._render(trainer, stage="batch")

    def on_fit_epoch_end(self, trainer: Any) -> None:
        self.epoch_rows.append(self._build_epoch_row(trainer))
        self._render(trainer, stage="epoch_end", force=True)

    def on_train_end(self, trainer: Any) -> None:
        self._render(trainer, stage="train_end", force=True)

    def _render(self, trainer: Any, *, stage: str, force: bool = False) -> None:
        now = time.perf_counter()
        should_render = force or self.batch_index <= 1
        should_render = should_render or (self.batch_index % self.every_n_batches == 0)
        should_render = should_render or ((now - self.last_render_at) >= self.refresh_seconds)
        if not should_render:
            return

        self.last_render_at = now
        text = self._build_text(trainer, stage=stage, now=now)
        self._display(text)

    def _build_text(self, trainer: Any, *, stage: str, now: float) -> str:
        elapsed = format_duration(now - self.started_at) if self.started_at else "0s"
        epoch_elapsed = format_duration(now - self.epoch_started_at) if self.epoch_started_at else "0s"
        gpu_mem = self._safe_gpu_memory(trainer)
        losses = self._loss_dict(trainer)
        metrics = self._metric_dict(getattr(trainer, "metrics", None))
        lr_values = self._metric_dict(getattr(trainer, "lr", None))

        stage_labels = {
            "starting": "initializing",
            "epoch_start": "epoch start",
            "batch": "training",
            "epoch_end": "epoch complete",
            "train_end": "finalizing",
        }
        stage_label = stage_labels.get(stage, stage)

        lines = [
            "MosKita Live Training Monitor",
            "=" * 72,
            f"Run      : {self.run_name}",
            f"Model    : {self.model_variant}",
            f"Status   : {stage_label}",
            (
                f"Progress : epoch {self.current_epoch}/{self.total_epochs or '?'}"
                f" | batch {self.batch_index}/{self.total_batches or '?'}"
            ),
            f"Elapsed  : {elapsed} total | {epoch_elapsed} this epoch | GPU reserved {gpu_mem}",
        ]

        if getattr(trainer, "batch_size", None) is not None:
            lines.append(f"Batch    : size {trainer.batch_size} | workers {getattr(getattr(trainer, 'train_loader', None), 'num_workers', '?')}")

        if losses:
            lines.append(
                "Losses   : "
                + " | ".join(f"{key.split('/', 1)[-1]} {value:.4f}" for key, value in losses.items())
            )

        if lr_values:
            lr_text = " | ".join(
                f"{key.split('/', 1)[-1]} {value:.2e}" for key, value in list(lr_values.items())[:3]
            )
            lines.append(f"LR       : {lr_text}")

        if metrics:
            summary_pairs = [
                ("precision", self._first_metric(metrics, "metrics/precision(B)", "metrics/precision")),
                ("recall", self._first_metric(metrics, "metrics/recall(B)", "metrics/recall")),
                ("mAP50", self._first_metric(metrics, "metrics/mAP50(B)", "metrics/mAP50")),
                ("mAP50-95", self._first_metric(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95")),
            ]
            summary_text = " | ".join(
                f"{label} {value:.4f}" for label, value in summary_pairs if value is not None
            )
            if summary_text:
                lines.append(f"Val      : {summary_text}")

        if self.epoch_rows:
            lines.extend(
                [
                    "",
                    "Recent Epochs",
                    "epoch  time      box      cls      dfl   precision   recall    mAP50  mAP50-95",
                ]
            )
            for row in self.epoch_rows[-self.recent_epoch_count :]:
                lines.append(
                    f"{int(row['epoch']):>5}  "
                    f"{format_duration(row['epoch_seconds'] or 0):<8}  "
                    f"{self._fmt_metric(row['box_loss']):>7}  "
                    f"{self._fmt_metric(row['cls_loss']):>7}  "
                    f"{self._fmt_metric(row['dfl_loss']):>7}  "
                    f"{self._fmt_metric(row['precision']):>10}  "
                    f"{self._fmt_metric(row['recall']):>7}  "
                    f"{self._fmt_metric(row['map50']):>8}  "
                    f"{self._fmt_metric(row['map50_95']):>9}"
                )

        return "\n".join(lines)

    def _display(self, text: str) -> None:
        if HTML is None or display is None:
            print(text)
            return

        payload = HTML(
            "<pre style='white-space:pre-wrap; line-height:1.35; "
            "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; "
            "padding:12px; border:1px solid #d0d7de; border-radius:8px; "
            "background:#f6f8fa;'>"
            f"{escape(text)}"
            "</pre>"
        )
        if self.display_handle is None:
            self.display_handle = display(payload, display_id=True)
            return
        try:
            self.display_handle.update(payload)
        except Exception:
            self.display_handle = display(payload, display_id=True)

    def _build_epoch_row(self, trainer: Any) -> dict[str, float | int | None]:
        losses = self._loss_dict(trainer)
        metrics = self._metric_dict(getattr(trainer, "metrics", None))
        epoch_seconds = getattr(trainer, "epoch_time", None)
        if epoch_seconds is None:
            epoch_seconds = time.perf_counter() - self.epoch_started_at

        return {
            "epoch": int(getattr(trainer, "epoch", 0) or 0) + 1,
            "epoch_seconds": float(epoch_seconds),
            "box_loss": losses.get("train/box_loss"),
            "cls_loss": losses.get("train/cls_loss"),
            "dfl_loss": losses.get("train/dfl_loss"),
            "precision": self._first_metric(metrics, "metrics/precision(B)", "metrics/precision"),
            "recall": self._first_metric(metrics, "metrics/recall(B)", "metrics/recall"),
            "map50": self._first_metric(metrics, "metrics/mAP50(B)", "metrics/mAP50"),
            "map50_95": self._first_metric(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
        }

    def _loss_dict(self, trainer: Any) -> dict[str, float]:
        label_loss_items = getattr(trainer, "label_loss_items", None)
        tloss = getattr(trainer, "tloss", None)
        if label_loss_items is None or tloss is None:
            return {}
        try:
            losses = label_loss_items(tloss, prefix="train")
        except Exception:
            return {}
        return self._metric_dict(losses)

    @staticmethod
    def _metric_dict(values: Any) -> dict[str, float]:
        if not isinstance(values, Mapping):
            return {}

        normalized: dict[str, float] = {}
        for key, value in values.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized

    @staticmethod
    def _first_metric(metrics: Mapping[str, float], *keys: str) -> float | None:
        for key in keys:
            if key in metrics:
                return float(metrics[key])
        return None

    @staticmethod
    def _fmt_metric(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    @staticmethod
    def _safe_total_batches(trainer: Any) -> int:
        train_loader = getattr(trainer, "train_loader", None)
        if train_loader is None:
            return 0
        try:
            return int(len(train_loader))
        except Exception:
            return 0

    @staticmethod
    def _safe_gpu_memory(trainer: Any) -> str:
        getter = getattr(trainer, "_get_memory", None)
        if getter is None:
            return "n/a"
        try:
            return f"{float(getter()):.2f}G"
        except Exception:
            return "n/a"
