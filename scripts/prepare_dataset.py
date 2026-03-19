from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test splits for the tooth numbering dataset."
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "yolo")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "tooth_numbers.yaml")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "data" / "dataset_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--class-prefix", default="tooth")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing processed dataset.")
    return parser


def discover_files(raw_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    image_dir = raw_dir / "images"
    label_dir = raw_dir / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Expected '{image_dir}' and '{label_dir}' to exist after extracting the dataset."
        )

    images = {
        path.stem: path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    labels = {
        path.stem: path
        for path in sorted(label_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    }
    return images, labels


def normalize_label_row(label_path: Path, line_number: int, raw_line: str) -> tuple[int, str]:
    line = raw_line.strip()
    parts = line.split()

    if len(parts) < 5:
        raise ValueError(f"Malformed label in {label_path} at line {line_number}: {raw_line!r}")

    class_id = int(parts[0])

    if len(parts) == 5:
        x_center, y_center, width, height = map(float, parts[1:])
        return class_id, f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"

    coordinate_values = [float(value) for value in parts[1:]]
    if len(coordinate_values) % 2 != 0:
        raise ValueError(f"Malformed polygon label in {label_path} at line {line_number}: {raw_line!r}")

    xs = coordinate_values[0::2]
    ys = coordinate_values[1::2]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return class_id, f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"


def normalize_label_file(label_path: Path) -> list[str]:
    normalized_rows: list[str] = []

    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            _, normalized_row = normalize_label_row(label_path, line_number, raw_line)
            normalized_rows.append(normalized_row)

    return normalized_rows


def summarize_labels(label_paths: list[Path]) -> tuple[dict[Path, list[str]], Counter[int], int]:
    normalized_labels: dict[Path, list[str]] = {}
    class_counts: Counter[int] = Counter()
    total_boxes = 0

    for label_path in label_paths:
        normalized_rows = normalize_label_file(label_path)
        normalized_labels[label_path] = normalized_rows
        for row in normalized_rows:
            class_id = int(row.split()[0])
            class_counts[class_id] += 1
            total_boxes += 1

    return normalized_labels, class_counts, total_boxes


def split_stems(stems: list[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[str]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1, and leave room for test.")

    shuffled = stems[:]
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)

    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def reset_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} already exists. Re-run with --force to replace it.")
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_split_files(
    splits: dict[str, list[str]],
    images: dict[str, Path],
    labels: dict[str, Path],
    normalized_labels: dict[Path, list[str]],
    output_dir: Path,
) -> None:
    for split_name, stems in splits.items():
        for stem in stems:
            shutil.copy2(images[stem], output_dir / split_name / "images" / images[stem].name)
            destination = output_dir / split_name / "labels" / labels[stem].name
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(normalized_labels[labels[stem]]))
                handle.write("\n")


def build_class_names(num_classes: int, prefix: str) -> list[str]:
    width = max(2, len(str(num_classes)))
    return [f"{prefix}_{index + 1:0{width}d}" for index in range(num_classes)]


def write_dataset_yaml(config_path: Path, output_dir: Path, num_classes: int, class_names: list[str]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset_path = output_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        dataset_path = str(output_dir)

    yaml_payload = {
        "path": dataset_path,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": num_classes,
        "names": class_names,
    }

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(yaml_payload, handle, sort_keys=False)


def write_report(
    report_path: Path,
    splits: dict[str, list[str]],
    class_counts: Counter[int],
    total_boxes: int,
    orphan_images: list[str],
    orphan_labels: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_images_used": sum(len(items) for items in splits.values()),
        "num_boxes": total_boxes,
        "split_sizes": {split: len(items) for split, items in splits.items()},
        "class_counts": {str(class_id): count for class_id, count in sorted(class_counts.items())},
        "orphan_images": orphan_images,
        "orphan_labels": orphan_labels,
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = build_parser().parse_args()

    images, labels = discover_files(args.raw_dir)
    paired_stems = sorted(images.keys() & labels.keys())
    orphan_images = sorted(images.keys() - labels.keys())
    orphan_labels = sorted(labels.keys() - images.keys())

    if not paired_stems:
        raise RuntimeError("No image/label pairs were found.")

    paired_label_paths = [labels[stem] for stem in paired_stems]
    normalized_labels, class_counts, total_boxes = summarize_labels(paired_label_paths)
    num_classes = max(class_counts) + 1
    class_names = build_class_names(num_classes, args.class_prefix)

    splits = split_stems(
        stems=paired_stems,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    reset_output_dir(args.output_dir, args.force)
    copy_split_files(splits, images, labels, normalized_labels, args.output_dir)
    write_dataset_yaml(args.config, args.output_dir, num_classes, class_names)
    write_report(args.report, splits, class_counts, total_boxes, orphan_images, orphan_labels)

    print(json.dumps(
        {
            "used_images": len(paired_stems),
            "split_sizes": {split: len(items) for split, items in splits.items()},
            "num_classes": num_classes,
            "total_boxes": total_boxes,
            "orphan_images": orphan_images,
            "orphan_labels": orphan_labels,
            "config": str(args.config),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
