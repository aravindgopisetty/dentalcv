# Tooth Number Detection with YOLO

This workspace is set up to train a YOLO detector on the dental tooth-numbering dataset in `data/raw`.

## Recommended model

`yolo11m.pt` is the recommended full-training starting point for Google Colab:

- It is meaningfully stronger than nano/small checkpoints for fine-grained tooth identity classification.
- It is still practical on a Colab T4 GPU for a 497-image dataset.
- The training script disables left/right flips and anatomy-breaking augmentations, which matters for tooth numbering.

For a quick local smoke test on CPU, use `yolo11n.pt`.

## Prepare the dataset

After extracting the zip into `data/raw`, build deterministic train/val/test splits and generate the dataset YAML:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_dataset.py --force
```

This creates:

- `data/yolo/train|val|test/...`
- `configs/tooth_numbers.yaml`
- `data/dataset_report.json`

## Local smoke test

This is only to verify the pipeline on CPU, not to get the best model:

```powershell
.\.venv\Scripts\python.exe .\scripts\train.py --model yolo11n.pt --epochs 1 --imgsz 640 --batch 4 --device cpu --name smoke-yolo11n --no-test
```

## Full Colab training

Upload this project folder plus `ToothNumber_TaskDataset.zip` to Colab, then run:

```python
!pip install ultralytics pyyaml
!mkdir -p data/raw
!unzip -q ToothNumber_TaskDataset.zip -d data/raw
!python scripts/prepare_dataset.py --force
!python scripts/train.py --model yolo11m.pt --epochs 120 --imgsz 960 --batch 8 --device 0 --cache --name tooth-yolo11m-1024
```

If GPU memory is tight, drop to `--imgsz 896` or `--model yolo11s.pt`.

There is also a ready-to-open notebook at `notebooks/tooth_yolo_colab.ipynb`.

To download the trained weights directly from Colab, use `scripts/download_model.py`.

## Outputs

Training artifacts are written under `runs/<run-name>/`, including:

- `weights/best.pt`
- `results.png`
- `confusion_matrix.png`
- `metrics_summary.json`

## Assumption

The dataset archive does not include class names, so the generated YAML uses placeholder names `tooth_01` through `tooth_32`. If you have a specific numbering convention, update the names list in `configs/tooth_numbers.yaml`.
