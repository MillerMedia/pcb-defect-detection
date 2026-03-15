# PCB Defect Detection Pipeline

**Author:** Matt Miller | **Course:** Computer Vision — Northwestern University

## Project Overview

This project implements a complete, reproducible end-to-end computer vision pipeline for detecting, localizing, and classifying printed circuit board (PCB) manufacturing defects. The pipeline combines multiple techniques — deep learning object detection, transfer-learning-based classification, and classical template matching — to provide a comprehensive analysis of defect detection approaches.

## Problem Statement

PCB manufacturing defects such as open circuits, shorts, mouse bites, spurs, spurious copper, and missing holes can cause costly failures in electronic devices. Automated visual inspection using computer vision can catch these defects early in the production process, reducing scrap rates and improving quality assurance.

## Techniques

| Approach | Model | Purpose |
|----------|-------|---------|
| Object Detection | YOLOv8 | Real-time defect localization and classification |
| Classification | ResNet-18/50 | Secondary validation on cropped detections |
| Template Matching | OpenCV | Classical baseline using paired reference images |

## Datasets

- **Kaggle PCB Defects** — 1,386 images with 6 defect classes (Pascal VOC annotations)
- **DeepPCB** — 1,500 paired template/defective images (used for template matching baseline)

## Pipeline Navigation

1. {doc}`notebooks/01-data-loading-and-eda` — Dataset download, format conversion, and exploratory data analysis
2. {doc}`notebooks/02-preprocessing-and-augmentation` — Image preprocessing and augmentation pipeline
3. {doc}`notebooks/03-yolov8-training-and-evaluation` — YOLOv8 defect detection training and evaluation
4. {doc}`notebooks/04-resnet-classification` — ResNet classification on cropped ROIs
5. {doc}`notebooks/05-template-matching-baseline` — Classical template matching baseline
6. {doc}`notebooks/06-model-comparison-and-analysis` — Cross-approach comparison and analysis
7. {doc}`notebooks/07-onnx-optimization` — ONNX export and quantization benchmarking
8. {doc}`notebooks/08-flask-inference-demo` — Flask inference API demonstration
