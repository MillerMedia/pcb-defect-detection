# PCB Defect Detection Pipeline

An end-to-end computer vision pipeline for detecting and classifying PCB manufacturing defects using YOLOv8, ResNet-18, and classical template matching.

**Author:** Matt Miller | Northwestern University | Computer Vision (COMP_SCI 449)

## Overview

This project compares three approaches to PCB defect detection across 6 defect types (missing hole, mouse bite, open circuit, short, spur, spurious copper):

| Approach | mAP@0.5 | Precision | Recall | F1 |
|---|---|---|---|---|
| YOLOv8 (detection) | **0.804** | **0.821** | **0.747** | **0.782** |
| ResNet-18 (GT crops) | N/A | 1.000 | 1.000 | 1.000 |
| Template Matching | N/A | 0.002 | 0.012 | 0.004 |

## Jupyter Book

Full documentation with executed notebooks, visualizations, and analysis:

**https://millermedia.github.io/pcb-defect-detection/**

## Pipeline

1. **Data Loading & EDA** — Kaggle PCB Defects (693 images) + DeepPCB (1,500 pairs)
2. **Preprocessing & Augmentation** — Letterbox resize, mosaic, flips, rotation, HSV jitter
3. **YOLOv8 Training** — YOLOv8n, COCO transfer learning, 70 epochs (early stopped at 60)
4. **ResNet Classification** — Two-stage pipeline: YOLOv8 detects → ResNet classifies crops
5. **Template Matching Baseline** — Classical CV comparison using DeepPCB paired images
6. **Model Comparison** — Cross-approach evaluation with per-class analysis
7. **ONNX Optimization** — INT8 quantization: 12MB → 3.5MB, quality gates passed
8. **Flask Inference API** — REST endpoint for real-time defect detection

## Project Structure

```
├── notebooks/
│   ├── 01-data-loading-and-eda.ipynb
│   ├── 02-preprocessing-and-augmentation.ipynb
│   ├── 03-yolov8-training-and-evaluation.ipynb
│   ├── 04-resnet-classification.ipynb
│   ├── 05-template-matching-baseline.ipynb
│   ├── 06-model-comparison-and-analysis.ipynb
│   ├── 07-onnx-optimization.ipynb
│   └── 08-flask-inference-demo.ipynb
├── app/
│   ├── app.py              # Flask inference API
│   └── utils.py            # Shared inference helpers
├── _config.yml             # Jupyter Book config
├── _toc.yml                # Jupyter Book table of contents
├── intro.md                # Jupyter Book landing page
└── requirements.txt        # Python dependencies
```

## Model Weights & Training Artifacts

Model weights (`.pt`, `.onnx`) and datasets are excluded from the repo via `.gitignore` due to file size. To reproduce:

1. **Download datasets** — Kaggle PCB Defects + DeepPCB (see Notebook 01)
2. **Run Notebook 03** to train YOLOv8 (saves to `models/yolov8_best.pt`)
3. **Run Notebook 07** to export ONNX models (saves to `models/yolov8_best.onnx` and `models/yolov8_best_int8.onnx`)

A full training log from the original 70-epoch run is included at `models/yolov8_pcb/training_log.txt` for reference. Training was performed on an Apple M4 Pro (MPS GPU) in approximately 2 hours with early stopping triggering at epoch 70 (best model at epoch 60).

## Quick Start

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run Flask API (after training or obtaining model weights)
cd app && MODEL_PATH=../models/yolov8_best.pt python app.py

# Build Jupyter Book
jupyter-book build .
```

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics), PyTorch
- **Classification:** ResNet-18 (torchvision)
- **Classical CV:** OpenCV (ORB, homography, template differencing)
- **Optimization:** ONNX Runtime, INT8 quantization
- **Deployment:** Flask REST API
- **Documentation:** Jupyter Book, GitHub Pages
