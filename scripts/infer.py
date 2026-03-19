from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLO tooth-numbering model.")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "artifacts" / "best.pt")
    parser.add_argument("--source", required=True, help="Image, directory, or glob pattern to run inference on.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None, help="Use '0' for GPU or 'cpu' for CPU inference.")
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--name", default="inference")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt predictions.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in txt outputs.")
    return parser


def pick_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.device)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        project=str(args.project),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )

    output_dir = Path(results[0].save_dir) if results else args.project / args.name
    print(f"Saved predictions to: {output_dir}")


if __name__ == "__main__":
    main()
