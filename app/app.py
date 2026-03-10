"""Flask inference API for PCB defect detection."""

import os
from flask import Flask, request, jsonify
from utils import load_model, preprocess_image, run_inference

app = Flask(__name__)

# Load model at startup
model_info = None


def get_model():
    global model_info
    if model_info is None:
        model_info = load_model()
    return model_info


@app.route("/health", methods=["GET"])
def health():
    m = get_model()
    if m is None:
        return jsonify({
            "status": "error",
            "message": "No model found. Set MODEL_PATH env var or place model in models/",
        }), 503

    return jsonify({
        "status": "healthy",
        "model_type": m["type"],
        "model_path": m["path"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    m = get_model()
    if m is None:
        return jsonify({"error": "No model loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as 'image' in multipart form data."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        img = preprocess_image(image_bytes)
        h, w = img.shape[:2]

        predictions, inference_time_ms = run_inference(img, m)

        return jsonify({
            "predictions": predictions,
            "image_size": [w, h],
            "inference_time_ms": round(inference_time_ms, 2),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("Loading model...")
    m = get_model()
    if m:
        print(f"Model loaded: {m['type']} from {m['path']}")
    else:
        print("WARNING: No model found. /predict will return 503.")

    app.run(host="0.0.0.0", port=5000, debug=False)
