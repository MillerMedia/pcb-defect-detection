# PCB Defect Detection: An End-to-End Multi-Technique Computer Vision Pipeline

**Matt Miller** | Northwestern University | COMP_SCI 449: Computer Vision | March 2026

## Problem Statement

PCB manufacturing requires high-throughput defect inspection where missed defects (false negatives) risk costly product failures while false positives waste rework time. This project builds a reproducible pipeline detecting six defect types — missing hole, mouse bite, open circuit, short, spur, and spurious copper — comparing classical template matching, YOLOv8 object detection, and a YOLOv8+ResNet two-stage pipeline, with ONNX export for deployment optimization.

## Datasets and Preprocessing

The **Kaggle PCB Defects** dataset provides 693 images with 2,953 bounding box annotations across 6 nearly balanced classes (imbalance ratio 1.04x). Images average 2778×2138 px. Pascal VOC XML annotations were converted to YOLO normalized format and split 70/15/15 stratified (485/104/104 images). **DeepPCB** provides 1,500 paired template/defective images reserved exclusively for template matching evaluation.

Images are letterbox-resized to 640×640 with aspect-preserving padding. YOLOv8 normalizes pixels to [0,1] internally; ResNet uses ImageNet normalization (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]). Augmentation includes mosaic, flips (horizontal + vertical), 90° rotation, HSV jitter, and scale variation — all PCB-appropriate since boards lack canonical orientation.

## Model Architectures

**Template Matching.** Given a defect-free template *T* and test image *I*, we align via ORB feature matching with RANSAC homography, compute *D(x,y)=|T_aligned(x,y)−I(x,y)|*, apply Gaussian blur and Otsu thresholding to produce a binary mask, then extract bounding boxes via contour detection (min area 50px²).

**YOLOv8.** Anchor-free single-stage detector with CSPDarknet53 backbone, PAN feature fusion neck, and decoupled classification/regression head. Loss combines CIoU (box), BCE (class), and distribution focal loss (DFL). Task-aligned assignment replaces IoU-based matching.

**ResNet-18.** Pretrained on ImageNet with final FC layer replaced (512→6 classes). Residual connections (*y=F(x,{Wᵢ})+x*) enable effective fine-tuning. In the two-stage pipeline, YOLOv8 detects regions → crops are classified by ResNet.

## Training Configuration

| | YOLOv8n | ResNet-18 |
|---|---|---|
| Epochs | 100 (patience=10) | 30 |
| Optimizer | AdamW, lr=0.01 | Adam, lr=1e-4 |
| Scheduler | — | CosineAnnealingLR |
| Input size | 640×640 | 224×224 (cropped ROIs) |
| Batch size | 16 | 32 |
| Frozen layers | — | conv1 through layer3 |
| Loss | CIoU+BCE+DFL | CrossEntropyLoss |

## Results and Evaluation

**Template matching baseline** evaluated on 1,500 DeepPCB pairs (IoU≥0.5): Precision=0.0023, Recall=0.0116, F1=0.0038. The near-zero precision reflects catastrophic false positive rates from alignment artifacts and lighting variation in pixel differencing. This establishes clear motivation for learned representations.

| Approach | mAP@0.5 | Precision | Recall | F1 | Inference (ms) |
|---|---|---|---|---|---|
| Template Matching | N/A | 0.002 | 0.012 | 0.004 | — |
| YOLOv8 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| ResNet (classification) | N/A | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| YOLOv8+ResNet (two-stage) | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

[PENDING: Per-class analysis, confusion matrix discussion, two-stage agreement analysis, ONNX optimization benchmarks (PyTorch vs ONNX vs INT8 — size, latency, quality gates ≤2% mAP drop, ≤1% recall drop).]

## Challenges and Solutions

**Format heterogeneity.** Kaggle (Pascal VOC XML) and DeepPCB (custom TXT with different class IDs) required dedicated converters to a canonical YOLO format with unified alphabetical class ordering, validated by visual spot-checks.

**Resolution scaling.** Source images (2778×2138) resized to 640×640 risk losing small defects. Letterbox padding preserves aspect ratio; multi-scale augmentation and the FPN neck partially mitigate detail loss.

**Template matching limitations.** Classical differencing fundamentally cannot discriminate defect types and requires per-design reference templates. Alignment failures on uniform PCB regions produce systematic false positives, confirming that learned features are essential for production-grade inspection.

[PENDING: Model-specific challenges from training — convergence behavior, class-specific failure modes, ONNX export considerations.]

---

*Code repository and Jupyter Book: [GitHub URL]*
