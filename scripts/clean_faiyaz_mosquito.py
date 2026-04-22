#!/usr/bin/env python3
"""
Remove Mosquito (class 1) and Mosquito Swarm (class 2) annotations from
the Faiyaz MosquitoFusion dataset.

Rules:
- Any label line starting with class 1 or 2 is removed.
- If a label file becomes empty after removal → delete both the .txt and its
  paired image.
- If a label file still has valid lines → rewrite it with only class-0 lines.

Supports --dry-run to preview without modifying anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
                    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

REMOVE_CLASSES = {1, 2}  # Mosquito, Mosquito Swarm


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def process_split(split_dir: Path, dry_run: bool) -> dict:
    label_dir = split_dir / "labels"
    image_dir = split_dir / "images"

    if not label_dir.exists():
        return {}

    stats = {
        "total_labels": 0,
        "labels_rewritten": 0,
        "labels_deleted": 0,
        "images_deleted": 0,
        "images_not_found": 0,
        "annotation_lines_removed": 0,
    }

    label_files = sorted(label_dir.glob("*.txt"))
    stats["total_labels"] = len(label_files)

    for lf in label_files:
        lines = lf.read_text().splitlines()
        kept = []
        removed_count = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            cls = int(stripped.split()[0])
            if cls in REMOVE_CLASSES:
                removed_count += 1
            else:
                kept.append(stripped)

        if removed_count == 0:
            continue  # nothing to do for this file

        stats["annotation_lines_removed"] += removed_count

        if kept:
            # Rewrite with surviving lines
            stats["labels_rewritten"] += 1
            if not dry_run:
                lf.write_text("\n".join(kept) + "\n")
        else:
            # No valid annotations left → delete label + image
            stats["labels_deleted"] += 1
            img = find_image(image_dir, lf.stem)
            if img:
                stats["images_deleted"] += 1
                if not dry_run:
                    img.unlink()
            else:
                stats["images_not_found"] += 1
                print(f"  [WARN] No image found for {lf.name}", file=sys.stderr)
            if not dry_run:
                lf.unlink()

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/annotated/outsource/faiyazabdullah/MosquitoFusion Dataset",
        help="Path to the MosquitoFusion dataset root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying any files.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    mode = "[DRY RUN]" if args.dry_run else "[LIVE]"
    print(f"{mode} Cleaning Faiyaz MosquitoFusion dataset")
    print(f"  Root : {root.resolve()}")
    print(f"  Removing classes : {REMOVE_CLASSES} (Mosquito, Mosquito Swarm)")
    print()

    total = {
        "total_labels": 0,
        "labels_rewritten": 0,
        "labels_deleted": 0,
        "images_deleted": 0,
        "images_not_found": 0,
        "annotation_lines_removed": 0,
    }

    for split in ("train", "valid", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        stats = process_split(split_dir, args.dry_run)
        if not stats:
            continue

        print(f"  [{split}]")
        print(f"    Label files scanned      : {stats['total_labels']}")
        print(f"    Annotation lines removed : {stats['annotation_lines_removed']}")
        print(f"    Label files rewritten    : {stats['labels_rewritten']}")
        print(f"    Label files deleted      : {stats['labels_deleted']}")
        print(f"    Images deleted           : {stats['images_deleted']}")
        if stats["images_not_found"]:
            print(f"    Images NOT found (warn)  : {stats['images_not_found']}")
        print()

        for k in total:
            total[k] += stats.get(k, 0)

    print("=" * 50)
    print(f"TOTAL annotation lines removed : {total['annotation_lines_removed']}")
    print(f"TOTAL label files rewritten    : {total['labels_rewritten']}")
    print(f"TOTAL label files deleted      : {total['labels_deleted']}")
    print(f"TOTAL images deleted           : {total['images_deleted']}")
    if total["images_not_found"]:
        print(f"TOTAL images NOT found (warn)  : {total['images_not_found']}")

    if args.dry_run:
        print()
        print("Dry run complete. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
