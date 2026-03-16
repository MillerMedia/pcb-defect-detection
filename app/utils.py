"""Shared inference helpers for PCB defect detection API."""

import os
import time
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}


def load_model(model_path=None):
    """Load inference model (ONNX or PyTorch).

    Fallback chain: ONNX → PyTorch .pt
    Both use the Ultralytics YOLO wrapper for consistent pre/post processing.
    """
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "models/yolov8_best.onnx")

    model_path = Path(model_path)

    if model_path.exists() and model_path.suffix in (".onnx", ".pt"):
        from ultralytics import YOLO
        model = YOLO(str(model_path), task="detect")
        return {"type": "ultralytics", "model": model, "path": str(model_path)}

    # Fallback to PyTorch if ONNX path given but not found
    pt_path = model_path.with_suffix(".pt")
    if not pt_path.exists():
        pt_path = Path("models/yolov8_best.pt")

    if pt_path.exists():
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        return {"type": "ultralytics", "model": model, "path": str(pt_path)}

    return None


def preprocess_image(image_bytes):
    """Decode and preprocess an image from bytes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def postprocess_predictions(results, model_info):
    """Format model predictions into API response format.

    Returns list of dicts with keys: class, confidence, bbox [x1, y1, x2, y2] in absolute pixels.
    """
    predictions = []

    if model_info["type"] == "ultralytics":
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            predictions.append({
                "class": CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
    elif model_info["type"] == "onnx":
        boxes, scores, class_ids = results
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i].tolist()
            predictions.append({
                "class": CLASS_NAMES.get(int(class_ids[i]), f"unknown_{int(class_ids[i])}"),
                "confidence": round(float(scores[i]), 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

    return predictions


def _onnx_inference(img, session):
    """Run inference using raw ONNX Runtime session with NMS post-processing."""
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    img_h, img_w = img.shape[:2]
    target_h, target_w = input_shape[2], input_shape[3]

    resized = cv2.resize(img, (target_w, target_h))
    blob = resized.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    outputs = session.run(None, {input_name: blob})
    preds = outputs[0][0].T  # (num_preds, 4+num_classes)

    scores = preds[:, 4:].max(axis=1)
    class_ids = preds[:, 4:].argmax(axis=1)
    mask = scores > 0.25
    preds, scores, class_ids = preds[mask], scores[mask], class_ids[mask]

    boxes = preds[:, :4].copy()
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w / target_w
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h / target_h
    boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] * img_w / target_w)
    boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] * img_h / target_h)

    return boxes, scores, class_ids


def run_inference(img, model_info):
    """Run inference and return predictions with timing."""
    start = time.time()

    if model_info["type"] == "ultralytics":
        results = model_info["model"].predict(img, verbose=False)
    elif model_info["type"] == "onnx":
        results = _onnx_inference(img, model_info["session"])

    elapsed_ms = (time.time() - start) * 1000
    predictions = postprocess_predictions(results, model_info)

    return predictions, elapsed_ms


def draw_predictions(img, predictions):
    """Draw bounding boxes and labels on image."""
    img_draw = img.copy()
    colors = {
        "missing_hole": (255, 0, 0),
        "mouse_bite": (0, 255, 0),
        "open_circuit": (0, 0, 255),
        "short": (255, 255, 0),
        "spur": (255, 0, 255),
        "spurious_copper": (0, 255, 255),
    }

    for pred in predictions:
        x1, y1, x2, y2 = [int(v) for v in pred["bbox"]]
        color = colors.get(pred["class"], (128, 128, 128))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        label = f"{pred['class']} {pred['confidence']:.2f}"
        cv2.putText(img_draw, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_draw
