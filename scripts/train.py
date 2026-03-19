from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO model for tooth numbering detection.")
    parser.add_argument("--model", default="yolo11m.pt", help="YOLO checkpoint to fine-tune.")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "configs" / "tooth_numbers.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Use '0' for Colab GPU or 'cpu' for local smoke tests.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--name", default="tooth-yolo11m")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="Skip final test-set evaluation.")
    return parser


def pick_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "0" if torch.cuda.is_available() else "cpu"


def metrics_to_dict(metrics: object) -> dict[str, float]:
    payload: dict[str, float] = {}

    if hasattr(metrics, "results_dict"):
        for key, value in getattr(metrics, "results_dict").items():
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                continue

    box = getattr(metrics, "box", None)
    if box is not None:
        for key in ("map", "map50", "map75"):
            value = getattr(box, key, None)
            if value is not None:
                payload[f"box_{key}"] = float(value)

    return payload


def ensure_test_split(data_yaml: Path) -> bool:
    with data_yaml.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return bool(payload.get("test"))


def save_summary(path: Path, summary: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.device)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        cache=args.cache,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        cos_lr=True,
        amp=device != "cpu",
        fliplr=0.0,
        flipud=0.0,
        degrees=3.0,
        translate=0.03,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        copy_paste=0.0,
        mixup=0.0,
        erasing=0.0,
        resume=args.resume,
        plots=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    best_model = YOLO(best_weights if best_weights.exists() else args.model)

    summary = {
        "model": args.model,
        "device": device,
        "save_dir": str(save_dir),
        "best_weights": str(best_weights),
    }

    val_metrics = best_model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        plots=True,
        project=str(args.project),
        name=f"{args.name}-val",
    )
    summary["val"] = metrics_to_dict(val_metrics)

    if not args.no_test and ensure_test_split(args.data):
        test_metrics = best_model.val(
            data=str(args.data),
            split="test",
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            plots=True,
            project=str(args.project),
            name=f"{args.name}-test",
        )
        summary["test"] = metrics_to_dict(test_metrics)

    save_summary(save_dir / "metrics_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
