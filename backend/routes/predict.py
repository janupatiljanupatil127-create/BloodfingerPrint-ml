"""
Predictor Utility
BloodPrint AI
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

random_forest = None
svm = None
knn = None
voting = None
scaler = None
label_encoder = None


def load_models():
    """
    Load all trained ML models.
    """

    global random_forest
    global svm
    global knn
    global voting
    global scaler
    global label_encoder

    model_files = {
        "Random Forest": "random_forest.pkl",
        "SVM": "svm.pkl",
        "KNN": "knn.pkl",
        "Voting": "voting.pkl",
        "Scaler": "scaler.pkl",
        "Label Encoder": "label_encoder.pkl"
    }

    loaded = {}

    for model_name, filename in model_files.items():

        file_path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"{filename} not found in {MODEL_DIR}"
            )

        loaded[model_name] = joblib.load(file_path)

    random_forest = loaded["Random Forest"]
    svm = loaded["SVM"]
    knn = loaded["KNN"]
    voting = loaded["Voting"]
    scaler = loaded["Scaler"]
    label_encoder = loaded["Label Encoder"]

    print("✅ All ML models loaded successfully.")

    return True


def models_loaded():
    """
    Returns True if all models are loaded.
    """

    return all([
        random_forest is not None,
        svm is not None,
        knn is not None,
        voting is not None,
        scaler is not None,
        label_encoder is not None
    ])


def predict(features):
    """
    Predict blood group using all trained models.
    """

    if not models_loaded():
        raise RuntimeError(
            "Models are not loaded. Run load_models() first."
        )

    sample = np.array(features).reshape(1, -1)

    sample = scaler.transform(sample)

    rf = label_encoder.inverse_transform(
        random_forest.predict(sample)
    )[0]

    svm_result = label_encoder.inverse_transform(
        svm.predict(sample)
    )[0]

    knn_result = label_encoder.inverse_transform(
        knn.predict(sample)
    )[0]

    voting_result = label_encoder.inverse_transform(
        voting.predict(sample)
    )[0]

    return {
        "random_forest": rf,
        "svm": svm_result,
        "knn": knn_result,
        "voting": voting_result
    }


def get_loaded_models():
    """
    Returns model loading status.
    """

    return {
        "Random Forest": random_forest is not None,
        "SVM": svm is not None,
        "KNN": knn is not None,
        "Voting Classifier": voting is not None,
        "Scaler": scaler is not None,
        "Label Encoder": label_encoder is not None
    }
