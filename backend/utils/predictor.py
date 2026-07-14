"""
Prediction module for BloodPrint AI
"""

import os
import joblib
import numpy as np

MODEL_DIR = "models"

rf = None
svm = None
knn = None
voting = None
scaler = None
encoder = None


def load_models():
    """
    Load trained models from disk.
    """

    global rf, svm, knn, voting, scaler, encoder

    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    svm = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
    knn = joblib.load(os.path.join(MODEL_DIR, "knn.pkl"))
    voting = joblib.load(os.path.join(MODEL_DIR, "voting.pkl"))

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


def predict(features):
    """
    Predict blood group using all models.
    """

    sample = scaler.transform([features])

    rf_pred = encoder.inverse_transform(
        rf.predict(sample)
    )[0]

    svm_pred = encoder.inverse_transform(
        svm.predict(sample)
    )[0]

    knn_pred = encoder.inverse_transform(
        knn.predict(sample)
    )[0]

    voting_pred = encoder.inverse_transform(
        voting.predict(sample)
    )[0]

    return {
        "random_forest": rf_pred,
        "svm": svm_pred,
        "knn": knn_pred,
        "voting": voting_pred
    }
