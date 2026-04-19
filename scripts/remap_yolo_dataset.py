#!/usr/bin/env python3
"""Remap YOLO class IDs by class-name mapping and merge into a unified dataset.

This tool is intended for YOLO detection workflows where source datasets may use
inconsistent class names and/or mixed annotation formats.

Supported input label rows:
- Detection: class x_center y_center width height
- Segmentation polygon: class x1 y1 x2 y2 ... xN yN (odd token count > 5)

When --convert-segments-to-boxes is enabled, polygon rows are converted to
axis-aligned detection boxes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".WEBP",
    ".TIF",
    ".TIFF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap YOLO labels by source class-name to target class-name."
    )
    parser.add_argument("--input-root", required=True, help="Source dataset root.")
    parser.add_argument(
        "--input-yaml",
        default=None,
        help="Path to source data.yaml (default: <input-root>/data.yaml).",
    )
    parser.add_argument("--output-root", required=True, help="Output merged dataset root.")
    parser.add_argument(
        "--class-map",
        required=True,
        help="Path to JSON mapping: {\"source_class\": \"target_class\", ...}",
    )
    parser.add_argument(
        "--target-names",
        required=True,
        help="Comma-separated target class names in final ID order.",
    )
    parser.add_argument(
        "--dataset-tag",
        default=None,
        help="Prefix for output filenames (default: sanitized source folder name).",
    )
    parser.add_argument(
        "--split-map",
        default="train:train,valid:val,val:val,test:test",
        help="Split remap pairs, comma-separated. Example: train:train,valid:val,test:test",
    )
    parser.add_argument(
        "--convert-segments-to-boxes",
        action="store_true",
        help="Convert polygon labels to detection boxes.",
    )
    parser.add_argument(
        "--drop-unmapped",
        action="store_true",
        help="Drop rows whose source class is not in --class-map.",
    )
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Keep images even when all rows were dropped (writes empty .txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only, no files written.",
    )
    parser.set_defaults(copy_images=True)
    parser.add_argument(
        "--no-copy-images",
        dest="copy_images",
        action="store_false",
        help="Do not copy image files (labels only).",
    )
    return parser.parse_args()


def sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "dataset"


def strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return value[:idx].rstrip()
    return value.rstrip()


def parse_scalar(value: str) -> str:
    value = strip_inline_comment(value).strip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            return str(ast.literal_eval(value))
        except Exception:
            return value[1:-1]
    return value


def load_source_names(data_yaml: Path) -> List[str]:
    lines = data_yaml.read_text(encoding="utf-8").splitlines()

    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped.startswith("names:"):
            continue

        names_indent = len(raw) - len(raw.lstrip(" "))
        after = stripped[len("names:") :].strip()

        if after:
            after = strip_inline_comment(after)
            parsed = ast.literal_eval(after)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            if isinstance(parsed, dict):
                return [str(parsed[k]) for k in sorted(parsed)]
            raise ValueError(f"Unsupported inline names format in {data_yaml}")

        list_names: List[str] = []
        dict_names: Dict[int, str] = {}

        for nxt in lines[i + 1 :]:
            if not nxt.strip() or nxt.strip().startswith("#"):
                continue
            indent = len(nxt) - len(nxt.lstrip(" "))
            if indent <= names_indent:
                break

            content = nxt.strip()
            if content.startswith("- "):
                list_names.append(parse_scalar(content[2:].strip()))
                continue

            match = re.match(r"^(\d+)\s*:\s*(.+)$", content)
            if match:
                dict_names[int(match.group(1))] = parse_scalar(match.group(2))
                continue

            raise ValueError(f"Unrecognized names entry '{content}' in {data_yaml}")

        if list_names and dict_names:
            raise ValueError(f"Mixed names styles not supported in {data_yaml}")
        if list_names:
            return list_names
        if dict_names:
            return [dict_names[k] for k in sorted(dict_names)]

        raise ValueError(f"Could not parse names block in {data_yaml}")

    raise ValueError(f"No 'names' block found in {data_yaml}")


def parse_target_names(value: str) -> List[str]:
    names = [x.strip() for x in value.split(",") if x.strip()]
    if not names:
        raise ValueError("--target-names produced an empty list")
    if len(names) != len(set(names)):
        raise ValueError("--target-names contains duplicate class names")
    return names


def load_class_map(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"class map must be a JSON object: {path}")

    result: Dict[str, str] = {}
    for k, v in payload.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"class map keys/values must be strings: {path}")
        result[k] = v
    return result


def parse_split_map(value: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid split mapping '{chunk}'")
        src, dst = chunk.split(":", 1)
        mapping[src.strip()] = dst.strip()

    if not mapping:
        raise ValueError("split map is empty")
    return mapping


def ensure_output_yaml(output_root: Path, target_names: Sequence[str], dry_run: bool) -> None:
    data_yaml = output_root / "data.yaml"

    if data_yaml.exists():
        existing_names = load_source_names(data_yaml)
        if existing_names != list(target_names):
            raise ValueError(
                "Existing output data.yaml class names do not match --target-names"
            )
        return

    lines = [
        "path: .",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        f"nc: {len(target_names)}",
        "names:",
    ]
    for idx, name in enumerate(target_names):
        lines.append(f"  {idx}: {name}")
    content = "\n".join(lines) + "\n"

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        data_yaml.write_text(content, encoding="utf-8")


def find_image_for_label(image_root: Path, label_rel_stem: Path) -> Optional[Path]:
    parent = image_root / label_rel_stem.parent
    stem = label_rel_stem.name

    for ext in IMAGE_EXTENSIONS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    if parent.exists():
        candidates = sorted(parent.glob(f"{stem}.*"))
        if candidates:
            return candidates[0]

    return None


def unique_label_path(directory: Path, stem: str) -> Path:
    candidate = directory / f"{stem}.txt"
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = directory / f"{stem}_{suffix}.txt"
        if not candidate.exists():
            return candidate
        suffix += 1


def convert_polygon_to_box(coords: Sequence[float]) -> Tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return x_center, y_center, width, height


def valid_box(x: float, y: float, w: float, h: float) -> bool:
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def remap_row(
    tokens: List[str],
    source_names: Sequence[str],
    class_map: Dict[str, str],
    target_to_id: Dict[str, int],
    convert_segments_to_boxes: bool,
) -> Tuple[Optional[str], str]:
    if not tokens:
        return None, "empty"

    if not tokens[0].isdigit():
        return None, "non_integer_class"

    source_id = int(tokens[0])
    if source_id < 0 or source_id >= len(source_names):
        return None, "source_id_out_of_range"

    source_name = source_names[source_id]
    if source_name not in class_map:
        return None, "unmapped_class"

    target_name = class_map[source_name]
    if target_name not in target_to_id:
        return None, "unknown_target_name"

    target_id = target_to_id[target_name]

    if len(tokens) == 5:
        try:
            x, y, w, h = map(float, tokens[1:])
        except ValueError:
            return None, "invalid_numeric"

    elif len(tokens) > 5 and len(tokens) % 2 == 1:
        if not convert_segments_to_boxes:
            return None, "segmentation_not_converted"

        try:
            coords = [float(v) for v in tokens[1:]]
        except ValueError:
            return None, "invalid_numeric"

        x, y, w, h = convert_polygon_to_box(coords)

    else:
        return None, "invalid_row_shape"

    if not valid_box(x, y, w, h):
        return None, "invalid_box_range"

    return f"{target_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}", "ok"


def main() -> int:
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    input_yaml = Path(args.input_yaml).resolve() if args.input_yaml else input_root / "data.yaml"

    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")
    if not input_yaml.exists():
        raise FileNotFoundError(f"source data.yaml not found: {input_yaml}")

    source_names = load_source_names(input_yaml)
    class_map = load_class_map(Path(args.class_map).resolve())
    target_names = parse_target_names(args.target_names)
    target_to_id = {name: idx for idx, name in enumerate(target_names)}
    split_map = parse_split_map(args.split_map)

    if args.dataset_tag:
        dataset_tag = sanitize_tag(args.dataset_tag)
    else:
        dataset_tag = sanitize_tag(input_root.name)

    ensure_output_yaml(output_root, target_names, args.dry_run)

    stats = Counter()
    row_rejections = Counter()

    for src_split, dst_split in split_map.items():
        src_labels = input_root / src_split / "labels"
        src_images = input_root / src_split / "images"
        if not src_labels.exists():
            continue

        stats["splits_processed"] += 1

        for label_file in sorted(src_labels.rglob("*.txt")):
            stats["label_files_seen"] += 1
            rel_stem = label_file.relative_to(src_labels).with_suffix("")
            src_image = find_image_for_label(src_images, rel_stem)

            if src_image is None:
                stats["files_skipped_missing_image"] += 1
                continue

            converted_rows: List[str] = []
            raw_lines = label_file.read_text(encoding="utf-8").splitlines()

            for raw_line in raw_lines:
                tokens = raw_line.strip().split()
                if not tokens:
                    continue

                row, reason = remap_row(
                    tokens=tokens,
                    source_names=source_names,
                    class_map=class_map,
                    target_to_id=target_to_id,
                    convert_segments_to_boxes=args.convert_segments_to_boxes,
                )

                if row is None:
                    row_rejections[reason] += 1
                    if reason == "unmapped_class" and not args.drop_unmapped:
                        raise RuntimeError(
                            "Encountered unmapped class while --drop-unmapped is off: "
                            f"{label_file} line='{raw_line}'"
                        )
                    continue

                converted_rows.append(row)
                stats["rows_written"] += 1

            if not converted_rows and not args.include_empty_labels:
                stats["files_skipped_empty_after_filter"] += 1
                continue

            base_name = f"{dataset_tag}_{src_split}_{str(rel_stem).replace('/', '__')}"
            dst_label_dir = output_root / dst_split / "labels"
            dst_image_dir = output_root / dst_split / "images"

            if not args.dry_run:
                dst_label_dir.mkdir(parents=True, exist_ok=True)
                dst_image_dir.mkdir(parents=True, exist_ok=True)

            dst_label = unique_label_path(dst_label_dir, base_name)
            dst_image = dst_image_dir / f"{dst_label.stem}{src_image.suffix.lower()}"

            if not args.dry_run:
                dst_label.write_text("\n".join(converted_rows) + ("\n" if converted_rows else ""), encoding="utf-8")
                if args.copy_images:
                    shutil.copy2(src_image, dst_image)

            stats["samples_written"] += 1

    print("=== Remap Summary ===")
    print(f"Input root: {input_root}")
    print(f"Input yaml: {input_yaml}")
    print(f"Output root: {output_root}")
    print(f"Dataset tag: {dataset_tag}")
    print(f"Source classes: {source_names}")
    print(f"Target classes: {target_names}")
    print(f"splits_processed={stats['splits_processed']}")
    print(f"label_files_seen={stats['label_files_seen']}")
    print(f"samples_written={stats['samples_written']}")
    print(f"rows_written={stats['rows_written']}")
    print(f"files_skipped_missing_image={stats['files_skipped_missing_image']}")
    print(f"files_skipped_empty_after_filter={stats['files_skipped_empty_after_filter']}")

    if row_rejections:
        print("row_rejections:")
        for key in sorted(row_rejections):
            print(f"  - {key}: {row_rejections[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
