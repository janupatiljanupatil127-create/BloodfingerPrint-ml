"""
BloodPrint AI
Train Machine Learning Models

Models:
1. Random Forest
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Voting Classifier
"""

import os
import cv2
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils.preprocess import preprocess_image
from utils.feature_extraction import extract_features


# Dataset path
DATASET_PATH = "dataset"

# Models path
MODEL_PATH = "models"


def load_dataset(dataset_path):
    """
    Load fingerprint images and extract features.
    """

    features = []
    labels = []

    print("Loading dataset...")

    for blood_group in sorted(os.listdir(dataset_path)):

        folder = os.path.join(dataset_path, blood_group)

        if not os.path.isdir(folder):
            continue

        print(f"Processing {blood_group}...")

        for file in os.listdir(folder):

            if not file.lower().endswith(".bmp"):
                continue

            image_path = os.path.join(folder, file)

            try:
                image = preprocess_image(image_path)
                feature = extract_features(image)

                features.append(feature)
                labels.append(blood_group)

            except Exception as e:
                print(f"Skipped {file}: {e}")

    return np.array(features), np.array(labels)

def main():

    # Load dataset
    X, y = load_dataset(DATASET_PATH)

    print(f"Total Images : {len(X)}")

    # Encode labels
    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    # Scale Features
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    print(
        "Random Forest Accuracy:",
        accuracy_score(y_test, rf_pred)
    )

    # -----------------------------
    # SVM
    # -----------------------------
    svm = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    svm.fit(X_train, y_train)

    svm_pred = svm.predict(X_test)

    print(
        "SVM Accuracy:",
        accuracy_score(y_test, svm_pred)
    )

    # -----------------------------
    # KNN
    # -----------------------------
    knn = KNeighborsClassifier(
        n_neighbors=5
    )

    knn.fit(X_train, y_train)

    knn_pred = knn.predict(X_test)

    print(
        "KNN Accuracy:",
        accuracy_score(y_test, knn_pred)
    )

    # -----------------------------
    # Voting Classifier
    # -----------------------------
    voting = VotingClassifier(

        estimators=[

            ("rf", rf),

            ("svm", svm),

            ("knn", knn)

        ],

        voting="hard"

    )

    voting.fit(X_train, y_train)

    voting_pred = voting.predict(X_test)

    print(
        "Voting Accuracy:",
        accuracy_score(y_test, voting_pred)
    )

    # Create models folder
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Save models
    joblib.dump(rf, os.path.join(MODEL_PATH, "random_forest.pkl"))

    joblib.dump(svm, os.path.join(MODEL_PATH, "svm.pkl"))

    joblib.dump(knn, os.path.join(MODEL_PATH, "knn.pkl"))

    joblib.dump(voting, os.path.join(MODEL_PATH, "voting.pkl"))

    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

    joblib.dump(encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))

    print("\nModels saved successfully.")


if __name__ == "__main__":
    main()
