from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download YOLO weights from a Colab run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to the run directory, e.g. runs/tooth-yolo11m-colab")
    parser.add_argument(
        "--file",
        choices=("best", "last", "zip"),
        default="best",
        help="Download best.pt, last.pt, or a zip containing both when available.",
    )
    parser.add_argument("--zip-name", default="tooth-model-files.zip", help="Name to use when --file zip is selected.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    weights_dir = run_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    try:
        from google.colab import files  # type: ignore
    except ImportError as exc:
        raise RuntimeError("This helper is intended to run inside Google Colab.") from exc

    if args.file == "best":
        if not best_path.exists():
            raise FileNotFoundError(f"Could not find {best_path}")
        files.download(str(best_path))
        return

    if args.file == "last":
        if not last_path.exists():
            raise FileNotFoundError(f"Could not find {last_path}")
        files.download(str(last_path))
        return

    zip_path = run_dir / args.zip_name
    with ZipFile(zip_path, "w") as archive:
        if best_path.exists():
            archive.write(best_path, arcname="best.pt")
        if last_path.exists():
            archive.write(last_path, arcname="last.pt")

    if not zip_path.exists():
        raise FileNotFoundError(f"Could not create {zip_path}")

    files.download(str(zip_path))


if __name__ == "__main__":
    main()
