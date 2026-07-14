from flask import Blueprint, request, jsonify
import os
import time

from utils.preprocess import preprocess_image
from utils.feature_extraction import extract_features
from utils.predictor import predict

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict_blood_group():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded."
        }), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({
            "success": False,
            "message": "No file selected."
        }), 400

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(upload_folder, image.filename)

    image.save(image_path)

    start = time.time()

    processed = preprocess_image(image_path)

    features = extract_features(processed)

    result = predict(features)

    end = time.time()

    result["prediction_time"] = round(end - start, 3)

    return jsonify({
        "success": True,
        "result": result
    })
