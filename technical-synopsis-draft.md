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
| Epochs | 70 (early stopped at 60) | 30 |
| Optimizer | AdamW, lr=0.01 | Adam, lr=1e-4 |
| Scheduler | — | CosineAnnealingLR |
| Input size | 640×640 | 224×224 (GT crops) |
| Batch size | 16 | 32 |
| Frozen layers | — | conv1 through layer3 |
| Loss | CIoU+BCE+DFL | CrossEntropyLoss |

## Results and Evaluation

| Approach | mAP@0.5 | Precision | Recall | F1 | Inference (ms) |
|---|---|---|---|---|---|
| Template Matching (DeepPCB) | N/A | 0.002 | 0.012 | 0.004 | — |
| YOLOv8 (detection) | **0.804** | **0.821** | **0.747** | **0.782** | 35.0 |
| ResNet-18 (GT crops) | N/A | 1.000 | 1.000 | 1.000 | — |
| YOLOv8+ResNet (two-stage) | 0.804 | 0.821 | 0.747 | 0.782 | ~79 |

**Template matching** achieved near-zero metrics (F1=0.004) due to catastrophic false positive rates from alignment artifacts and lighting variation, establishing clear motivation for learned representations.

**YOLOv8** achieved mAP@0.5=0.804 on the test set with strong per-class variation: missing_hole (0.962 AP) was near-perfect while spur (0.660) and mouse_bite (0.672) proved hardest — these are small, subtle defects that also challenge human inspectors. Early stopping triggered at epoch 70 (best model at epoch 60), confirming convergence without overfitting.

**ResNet-18** achieved 100% classification accuracy on ground-truth crops, but this reflects perfect localization input. The **two-stage pipeline** (YOLOv8→ResNet) showed 98.1% class agreement across 426 detections, adding ~44ms latency with negligible accuracy gain — single-pass YOLOv8 is recommended for production.

**ONNX optimization** exported the model for runtime-agnostic deployment. INT8 quantization compressed the model from 12.0MB to 3.5MB (3.5× reduction) with quality gates passing: mAP drop of 0.48% (threshold ≤2%) and recall actually improved by 1.1%. Inference: PyTorch 35ms → ONNX FP32 28ms (19% faster) → INT8 31ms.

## Challenges and Solutions

**Format heterogeneity.** Kaggle (Pascal VOC XML) and DeepPCB (custom TXT with different class IDs) required dedicated converters to a canonical YOLO format with unified alphabetical class ordering, validated by visual spot-checks.

**Resolution scaling.** Source images (2778×2138) resized to 640×640 risk losing small defects. Letterbox padding preserves aspect ratio; multi-scale augmentation and the FPN neck partially mitigate detail loss.

**Class-specific difficulty.** Spur (AP=0.660) and mouse_bite (AP=0.672) defects are visually subtle and small, often missed or confused with background. Improvement paths include upgrading to YOLOv8s/m, collecting more training data for weak classes, and lowering the confidence threshold to favor recall over precision given the 10× cost asymmetry of missed defects.

**Single board design.** All training images derive from one PCB layout with synthetic defects. Production deployment requires per-board fine-tuning, though transfer learning from this model would reduce data requirements to ~100-200 labeled images per new design.

---

*Code repository: [github.com/MillerMedia/pcb-defect-detection](https://github.com/MillerMedia/pcb-defect-detection) | Jupyter Book: [millermedia.github.io/pcb-defect-detection](https://millermedia.github.io/pcb-defect-detection)*
