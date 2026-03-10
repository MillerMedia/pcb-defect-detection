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
    """
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "models/yolov8_best.onnx")

    model_path = Path(model_path)

    if model_path.suffix == ".onnx" and model_path.exists():
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        return {"type": "onnx", "session": session, "path": str(model_path)}

    # Fallback to PyTorch
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

    return predictions


def run_inference(img, model_info):
    """Run inference and return predictions with timing."""
    start = time.time()

    if model_info["type"] == "ultralytics":
        results = model_info["model"].predict(img, verbose=False)
    elif model_info["type"] == "onnx":
        # For ONNX, use Ultralytics to handle pre/post processing
        from ultralytics import YOLO
        model = YOLO(model_info["path"])
        results = model.predict(img, verbose=False)

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
