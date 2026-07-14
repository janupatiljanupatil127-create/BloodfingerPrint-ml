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
    print("Training pipeline initialized...")
    print("Dataset path:", DATASET_PATH)


if __name__ == "__main__":
    main()
