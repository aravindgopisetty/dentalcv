from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]


def add_cover_page(pdf: PdfPages, dataset: dict, metrics: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, "Tooth Number Detection with YOLO11", fontsize=22, fontweight="bold")
    fig.text(0.08, 0.915, "Short project report", fontsize=12, color="#555555")

    left_x = 0.08
    y = 0.86
    line_gap = 0.032

    lines = [
        "Objective",
        "Train a YOLO-based detector that identifies and localizes 32 tooth-number classes in panoramic dental images.",
        "",
        "Dataset",
        f"- Images used: {dataset['num_images_used']}",
        f"- Bounding boxes: {dataset['num_boxes']}",
        f"- Split: train {dataset['split_sizes']['train']}, val {dataset['split_sizes']['val']}, test {dataset['split_sizes']['test']}",
        "",
        "Approach",
        "- Mixed YOLO box and polygon annotations were normalized into detection boxes.",
        "- Ultralytics YOLO11m was fine-tuned with transfer learning.",
        "- Input size was increased to 1024 to improve tight localization on dense tooth structures.",
        "- Flips and anatomy-breaking augmentations were disabled because tooth numbering depends on left-right consistency.",
        "",
        "Training setup",
        "- Model: yolo11m.pt",
        "- Epochs: 120",
        "- Image size: 1024",
        "- Batch size: 6",
        "- Optimizer: AdamW",
        "",
        "Final metrics",
        f"- Validation precision / recall: {metrics['val']['metrics/precision(B)']:.4f} / {metrics['val']['metrics/recall(B)']:.4f}",
        f"- Validation mAP50 / mAP50-95: {metrics['val']['metrics/mAP50(B)']:.4f} / {metrics['val']['metrics/mAP50-95(B)']:.4f}",
        f"- Test precision / recall: {metrics['test']['metrics/precision(B)']:.4f} / {metrics['test']['metrics/recall(B)']:.4f}",
        f"- Test mAP50 / mAP50-95: {metrics['test']['metrics/mAP50(B)']:.4f} / {metrics['test']['metrics/mAP50-95(B)']:.4f}",
    ]

    for line in lines:
        if not line:
            y -= line_gap * 0.6
            continue
        weight = "bold" if line in {"Objective", "Dataset", "Approach", "Training setup", "Final metrics"} else "normal"
        fontsize = 13 if weight == "bold" else 11
        fig.text(left_x, y, line, fontsize=fontsize, fontweight=weight, va="top")
        y -= line_gap

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_full_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.88])
    ax.imshow(plt.imread(image_path))
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_two_image_page(pdf: PdfPages, title: str, image_a: Path, caption_a: str, image_b: Path, caption_b: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    for ax, image_path, caption in (
        (axes[0], image_a, caption_a),
        (axes[1], image_b, caption_b),
    ):
        ax.imshow(plt.imread(image_path))
        ax.axis("off")
        ax.set_title(caption, fontsize=12, pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dataset = json.loads((REPO_ROOT / "data" / "dataset_report.json").read_text(encoding="utf-8"))
    metrics = json.loads((REPO_ROOT / "artifacts" / "metrics_summary.json").read_text(encoding="utf-8"))

    training_curves = REPO_ROOT / "assets" / "training_results.png"
    confusion_matrix = REPO_ROOT / "assets" / "confusion_matrix.png"
    sample_source = REPO_ROOT / "sample_outputs" / "source" / "0ba65172-20240821-105924223.jpg"
    sample_prediction = REPO_ROOT / "sample_outputs" / "predictions" / "0ba65172-20240821-105924223_pred.jpg"

    output_dir = REPO_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "approach_report.pdf"

    with PdfPages(output_pdf) as pdf:
        add_cover_page(pdf, dataset, metrics)
        add_full_image_page(pdf, "Training curves", training_curves)
        add_two_image_page(
            pdf,
            "Confusion matrix and sample inference",
            confusion_matrix,
            "Confusion matrix",
            sample_prediction,
            "Sample prediction output",
        )
        add_two_image_page(
            pdf,
            "Input image and sample prediction",
            sample_source,
            "Input panoramic image",
            sample_prediction,
            "Predicted tooth numbering output",
        )

    print(f"Saved report to: {output_pdf}")


if __name__ == "__main__":
    main()
