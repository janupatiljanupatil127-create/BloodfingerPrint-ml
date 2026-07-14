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
    Load all trained models.
    """

    global random_forest
    global svm
    global knn
    global voting
    global scaler
    global label_encoder

    random_forest = joblib.load(
        os.path.join(MODEL_DIR, "random_forest.pkl")
    )

    svm = joblib.load(
        os.path.join(MODEL_DIR, "svm.pkl")
    )

    knn = joblib.load(
        os.path.join(MODEL_DIR, "knn.pkl")
    )

    voting = joblib.load(
        os.path.join(MODEL_DIR, "voting.pkl")
    )

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    label_encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    print("All models loaded successfully.")


def predict(features):
    """
    Predict blood group using all models.
    """

    sample = np.array(features).reshape(1, -1)

    sample = scaler.transform(sample)

    rf_prediction = label_encoder.inverse_transform(
        random_forest.predict(sample)
    )[0]

    svm_prediction = label_encoder.inverse_transform(
        svm.predict(sample)
    )[0]

    knn_prediction = label_encoder.inverse_transform(
        knn.predict(sample)
    )[0]

    voting_prediction = label_encoder.inverse_transform(
        voting.predict(sample)
    )[0]

    return {

        "random_forest": rf_prediction,

        "svm": svm_prediction,

        "knn": knn_prediction,

        "voting": voting_prediction

    }


def models_loaded():
    """
    Check whether models are loaded.
    """

    return all([
        random_forest,
        svm,
        knn,
        voting,
        scaler,
        label_encoder
    ])
