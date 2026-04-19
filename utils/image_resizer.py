#!/usr/bin/env python3
"""Simple image prep utility for MosKita.

This script reads raw images from INPUT_ROOT, resizes each image to a square
size for YOLO, renames files with a standard sequential pattern, and writes all
outputs directly into OUTPUT_ROOT.
"""

from __future__ import annotations

import csv
import os
import re
import time
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
OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw"
TARGET_SIZE = 1280  # Allowed values: 640 or 1280
FILENAME_PREFIX = "moskita"
ZERO_PADDING = 4
OUTPUT_FORMAT = "jpg"  # Allowed values: jpg or png
VERBOSE = True
SCAN_LOG_EVERY = 1000


def log(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


def sanitize_folder_label(folder_name: str) -> str:
    folder_name = folder_name.strip().lower()
    folder_name = re.sub(r"[^a-z0-9]+", "_", folder_name)
    folder_name = re.sub(r"_+", "_", folder_name).strip("_")
    return folder_name or "raw"


def folder_label_for_path(source_path: Path, input_root: Path) -> str:
    relative_path = source_path.relative_to(input_root)
    if not relative_path.parts:
        return "raw"
    return sanitize_folder_label(relative_path.parts[0])


def collect_images(input_root: Path) -> list[Path]:
    image_paths: list[Path] = []
    scanned_paths = 0

    for path in input_root.rglob("*"):
        scanned_paths += 1
        if VERBOSE and scanned_paths % SCAN_LOG_EVERY == 0:
            log(f"[SCAN] Checked {scanned_paths:,} paths, found {len(image_paths):,} candidate images...")

        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name == ".gitkeep":
            continue
        if path.stem.lower().startswith(f"{FILENAME_PREFIX}_"):
            continue
        image_paths.append(path)

    log(f"[SCAN] Completed. Checked {scanned_paths:,} paths, found {len(image_paths):,} candidate images.")
    return sorted(image_paths)


def collect_existing_indices(input_root: Path) -> dict[str, int]:
    highest_index: dict[str, int] = defaultdict(int)
    scanned_paths = 0

    for path in input_root.rglob("*"):
        scanned_paths += 1
        if VERBOSE and scanned_paths % SCAN_LOG_EVERY == 0:
            log(f"[INDEX] Checked {scanned_paths:,} paths for existing index values...")

        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        match = re.fullmatch(rf"{FILENAME_PREFIX}_(.+)_(\d+)", path.stem.lower())
        if not match:
            continue

        folder_label = sanitize_folder_label(match.group(1))
        index_value = int(match.group(2))
        if index_value > highest_index[folder_label]:
            highest_index[folder_label] = index_value

    log(
        f"[INDEX] Completed. Checked {scanned_paths:,} paths and tracked "
        f"{len(highest_index):,} folder label(s)."
    )
    return highest_index


def resize_and_save(source_path: Path, output_path: Path, size: int, output_format: str) -> tuple[int, int]:
    with Image.open(source_path) as image:
        original_width, original_height = image.size
        if original_width != original_height:
            raise ValueError(f"Image is not 1:1: {original_width}x{original_height}")

        resized = image.convert("RGB").resize((size, size), resample=Image.Resampling.LANCZOS)

        # Write using a file handle so we can flush + fsync immediately.
        with output_path.open("wb") as output_file:
            if output_format == "jpg":
                resized.save(output_file, format="JPEG", quality=95, optimize=True)
            else:
                resized.save(output_file, format="PNG", optimize=True)

            output_file.flush()
            os.fsync(output_file.fileno())

        return original_width, original_height


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
    start_time = time.time()
    log("[START] MosKita image resizer started.")

    validate_config()
    log(f"[CONFIG] Input root: {INPUT_ROOT}")
    log(f"[CONFIG] Output root: {OUTPUT_ROOT}")
    log(f"[CONFIG] Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    log(f"[CONFIG] Output format: {OUTPUT_FORMAT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log("[STEP] Scanning for source images...")
    image_paths = collect_images(INPUT_ROOT)

    if not image_paths:
        raise SystemExit(f"No supported image files found under: {INPUT_ROOT}")

    log("[STEP] Detecting existing moskita indices...")
    existing_indices = collect_existing_indices(INPUT_ROOT)
    folder_totals: dict[str, int] = defaultdict(int)
    for source_path in image_paths:
        folder_label = folder_label_for_path(source_path, INPUT_ROOT)
        folder_totals[folder_label] += 1

    next_indices = {
        folder_label: existing_index + 1
        for folder_label, existing_index in existing_indices.items()
    }

    log_path = OUTPUT_ROOT / "conversion_log.csv"
    processed_count = 0
    skipped = 0
    extension = "jpg" if OUTPUT_FORMAT == "jpg" else "png"
    total_images = len(image_paths)

    log(f"[STEP] Starting resize loop for {total_images:,} image(s)...")
    interrupted = False

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
        csv_file.flush()
        os.fsync(csv_file.fileno())

        try:
            for index, source_path in enumerate(image_paths, start=1):
                output_dir = OUTPUT_ROOT
                output_dir.mkdir(parents=True, exist_ok=True)

                folder_label = folder_label_for_path(source_path, INPUT_ROOT)
                current_index = next_indices.get(folder_label, 1)
                max_index_for_width = max(
                    existing_indices.get(folder_label, 0) + folder_totals[folder_label],
                    current_index,
                )
                width = max(ZERO_PADDING, len(str(max_index_for_width)))

                output_name = f"{FILENAME_PREFIX}_{folder_label}_{current_index:0{width}d}.{extension}"
                output_path = output_dir / output_name

                while output_path.exists():
                    current_index += 1
                    width = max(ZERO_PADDING, len(str(current_index)))
                    output_name = f"{FILENAME_PREFIX}_{folder_label}_{current_index:0{width}d}.{extension}"
                    output_path = output_dir / output_name

                source_rel = source_path.relative_to(INPUT_ROOT)
                output_rel = output_path.relative_to(PROJECT_ROOT)
                log(
                    f"[PROCESS {index:,}/{total_images:,}] {source_rel} -> {output_rel}"
                )

                try:
                    original_width, original_height = resize_and_save(
                        source_path=source_path,
                        output_path=output_path,
                        size=TARGET_SIZE,
                        output_format=OUTPUT_FORMAT,
                    )

                    next_indices[folder_label] = current_index + 1

                    writer.writerow(
                        (
                            output_name,
                            str(output_path),
                            str(source_path),
                            original_width,
                            original_height,
                            TARGET_SIZE,
                        )
                    )
                    csv_file.flush()
                    os.fsync(csv_file.fileno())

                    processed_count += 1
                    log(
                        f"[OK {index:,}/{total_images:,}] saved {output_name} "
                        f"from {original_width}x{original_height}"
                    )
                except Exception as exc:  # pragma: no cover - data-dependent failures
                    skipped += 1
                    print(f"[WARN {index:,}/{total_images:,}] Skipped {source_path}: {exc}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[STOP] Interrupted by user. Finalizing partial summary...", flush=True)

    if processed_count == 0:
        raise SystemExit("No images were successfully processed.")

    elapsed_seconds = time.time() - start_time
    status = "INTERRUPTED" if interrupted else "DONE"

    print(f"[{status}] Input root:  {INPUT_ROOT}")
    print(f"[{status}] Output root: {OUTPUT_ROOT}")
    print(f"[{status}] Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"[{status}] Processed:   {processed_count} image(s)")
    print(f"[{status}] Skipped:     {skipped} image(s)")
    print(f"[{status}] Elapsed:     {elapsed_seconds:.2f}s")
    print(f"[{status}] Log written: {log_path}")


if __name__ == "__main__":
    main()
