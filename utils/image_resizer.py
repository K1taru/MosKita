#!/usr/bin/env python3
"""Simple image prep utility for MosKita.

This script reads raw images from INPUT_ROOT, resizes each image to a square
size for YOLO, renames files with a standard sequential pattern, and writes the
outputs into OUTPUT_ROOT while preserving the same folder structure.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("Missing dependency: Pillow. Install it with pip install pillow.") from exc

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Update these constants when needed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "k1taru"
TARGET_SIZE = 640  # Allowed values: 640 or 1280
FILENAME_PREFIX = "moskita_drum"
ZERO_PADDING = 4
OUTPUT_FORMAT = "jpg"  # Allowed values: jpg or png


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def collect_images(input_root: Path, output_root: Path) -> list[Path]:
    image_paths: list[Path] = []
    for path in input_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if is_inside(path, output_root):
            continue
        image_paths.append(path)
    return sorted(image_paths)


def resize_and_save(source_path: Path, output_path: Path, size: int, output_format: str) -> tuple[int, int]:
    with Image.open(source_path) as image:
        original_width, original_height = image.size
        if original_width != original_height:
            raise ValueError(f"Image is not 1:1: {original_width}x{original_height}")

        resized = image.convert("RGB").resize((size, size), resample=Image.Resampling.LANCZOS)

        if output_format == "jpg":
            resized.save(output_path, format="JPEG", quality=95, optimize=True)
        else:
            resized.save(output_path, format="PNG", optimize=True)

        return original_width, original_height


def write_log(rows: list[tuple[str, str, str, int, int, int]], log_path: Path) -> None:
    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "new_filename",
                "output_path",
                "original_path",
                "original_width",
                "original_height",
                "target_size",
            ]
        )
        writer.writerows(rows)


def validate_config() -> None:
    if TARGET_SIZE not in {640, 1280}:
        raise SystemExit("TARGET_SIZE must be 640 or 1280.")
    if OUTPUT_FORMAT not in {"jpg", "png"}:
        raise SystemExit("OUTPUT_FORMAT must be jpg or png.")
    if ZERO_PADDING < 1:
        raise SystemExit("ZERO_PADDING must be >= 1.")
    if not INPUT_ROOT.exists() or not INPUT_ROOT.is_dir():
        raise SystemExit(f"Input directory not found: {INPUT_ROOT}")


def main() -> None:
    validate_config()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(INPUT_ROOT, OUTPUT_ROOT)

    if not image_paths:
        raise SystemExit(f"No supported image files found under: {INPUT_ROOT}")

    folder_totals: dict[Path, int] = defaultdict(int)
    for source_path in image_paths:
        relative_folder = source_path.parent.relative_to(INPUT_ROOT)
        folder_totals[relative_folder] += 1

    folder_indices: dict[Path, int] = defaultdict(int)
    processed_rows: list[tuple[str, str, str, int, int, int]] = []
    skipped = 0
    extension = "jpg" if OUTPUT_FORMAT == "jpg" else "png"

    for source_path in image_paths:
        relative_folder = source_path.parent.relative_to(INPUT_ROOT)
        output_dir = OUTPUT_ROOT / relative_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        folder_indices[relative_folder] += 1
        current_index = folder_indices[relative_folder]
        width = max(ZERO_PADDING, len(str(folder_totals[relative_folder])))
        output_name = f"{FILENAME_PREFIX}_{current_index:0{width}d}.{extension}"
        output_path = output_dir / output_name

        try:
            original_width, original_height = resize_and_save(
                source_path=source_path,
                output_path=output_path,
                size=TARGET_SIZE,
                output_format=OUTPUT_FORMAT,
            )
            processed_rows.append(
                (
                    output_name,
                    str(output_path),
                    str(source_path),
                    original_width,
                    original_height,
                    TARGET_SIZE,
                )
            )
        except Exception as exc:  # pragma: no cover - data-dependent failures
            skipped += 1
            print(f"[WARN] Skipped {source_path}: {exc}")

    if not processed_rows:
        raise SystemExit("No images were successfully processed.")

    log_path = OUTPUT_ROOT / "conversion_log.csv"
    write_log(rows=processed_rows, log_path=log_path)

    print(f"Input root:  {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Processed:   {len(processed_rows)} image(s)")
    print(f"Skipped:     {skipped} image(s)")
    print(f"Log written: {log_path}")


if __name__ == "__main__":
    main()
