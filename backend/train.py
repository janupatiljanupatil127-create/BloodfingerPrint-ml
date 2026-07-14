"""
=========================================================
 BloodPrint AI
 Fingerprint Based Blood Group Prediction

 Models
 -------
 1. Random Forest
 2. Support Vector Machine (SVM)
 3. K-Nearest Neighbors (KNN)
 4. Voting Classifier

 Author : Namrata Nagraj
=========================================================
"""

import os
import joblib
import warnings
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils.preprocess import preprocess_image
from utils.feature_extraction import extract_features

warnings.filterwarnings("ignore")

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_DIR = os.path.join(BASE_DIR, "models")

IMAGE_EXTENSIONS = (
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg"
)

# =====================================================
# Dataset Loader
# =====================================================

def load_dataset():

    print("\nLoading Fingerprint Dataset...\n")

    features = []

    labels = []

    classes = sorted(os.listdir(DATASET_DIR))

    print("Blood Groups Found")

    print(classes)

    print()

    total_images = 0

    for blood_group in classes:

        folder = os.path.join(
            DATASET_DIR,
            blood_group
        )

        if not os.path.isdir(folder):
            continue

        image_files = [

            file

            for file in os.listdir(folder)

            if file.lower().endswith(
                IMAGE_EXTENSIONS
            )

        ]

        print(
            f"{blood_group} -> {len(image_files)} Images"
        )

        for image_name in tqdm(
            image_files,
            desc=blood_group
        ):

            image_path = os.path.join(
                folder,
                image_name
            )

            try:

                image = preprocess_image(
                    image_path
                )

                feature_vector = extract_features(
                    image
                )

                features.append(
                    feature_vector
                )

                labels.append(
                    blood_group
                )

                total_images += 1

            except Exception as error:

                print(
                    f"Skipped {image_name}"
                )

                print(error)

    print()

    print("=" * 60)

    print(
        f"Total Images Loaded : {total_images}"
    )

    print("=" * 60)

    return (
        np.array(features),
        np.array(labels)
    )
# =====================================================
# Data Preparation
# =====================================================

def prepare_data():

    print("\nPreparing Dataset...\n")

    X, y = load_dataset()

    # Encode Labels
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    print("Label Encoding Completed")

    # Scale Features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("Feature Scaling Completed")

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(

        X_scaled,

        y_encoded,

        test_size=0.20,

        random_state=42,

        stratify=y_encoded

    )

    print()

    print("=" * 60)

    print("Dataset Summary")

    print("=" * 60)

    print(f"Training Samples : {len(X_train)}")

    print(f"Testing Samples  : {len(X_test)}")

    print(f"Feature Length   : {X.shape[1]}")

    print("=" * 60)

    return (

        X_train,

        X_test,

        y_train,

        y_test,

        scaler,

        encoder

    )
    # =====================================================
# Model Training
# =====================================================

def train_models():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder
    ) = prepare_data()

    print("\nTraining Machine Learning Models...\n")

    # -------------------------------------------------
    # Random Forest
    # -------------------------------------------------

    print("=" * 60)
    print("Training Random Forest...")
    print("=" * 60)

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(X_train, y_train)

    rf_prediction = random_forest.predict(X_test)

    rf_accuracy = accuracy_score(
        y_test,
        rf_prediction
    )

    print(f"Random Forest Accuracy : {rf_accuracy:.4f}")

    # -------------------------------------------------
    # Support Vector Machine
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("Training Support Vector Machine...")
    print("=" * 60)

    svm = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    svm.fit(X_train, y_train)

    svm_prediction = svm.predict(X_test)

    svm_accuracy = accuracy_score(
        y_test,
        svm_prediction
    )

    print(f"SVM Accuracy : {svm_accuracy:.4f}")

    # -------------------------------------------------
    # K-Nearest Neighbors
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("Training KNN...")
    print("=" * 60)

    knn = KNeighborsClassifier(
        n_neighbors=5
    )

    knn.fit(X_train, y_train)

    knn_prediction = knn.predict(X_test)

    knn_accuracy = accuracy_score(
        y_test,
        knn_prediction
    )

    print(f"KNN Accuracy : {knn_accuracy:.4f}")

    return (
        random_forest,
        svm,
        knn,
        scaler,
        encoder,
        X_train,
        X_test,
        y_train,
        y_test
    )
    # =====================================================
# Voting Classifier
# =====================================================

def train_voting_classifier():

    (
        random_forest,
        svm,
        knn,
        scaler,
        encoder,
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_models()

    print("\n" + "=" * 60)
    print("Training Voting Classifier...")
    print("=" * 60)

    voting = VotingClassifier(

        estimators=[

            ("rf", random_forest),

            ("svm", svm),

            ("knn", knn)

        ],

        voting="hard"

    )

    voting.fit(X_train, y_train)

    voting_prediction = voting.predict(X_test)

    voting_accuracy = accuracy_score(
        y_test,
        voting_prediction
    )

    print(f"\nVoting Classifier Accuracy : {voting_accuracy:.4f}")

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            voting_prediction
        )
    )

    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)

    print(
        confusion_matrix(
            y_test,
            voting_prediction
        )
    )

    print("\n" + "=" * 60)
    print("Model Accuracy Summary")
    print("=" * 60)

    rf_accuracy = accuracy_score(
        y_test,
        random_forest.predict(X_test)
    )

    svm_accuracy = accuracy_score(
        y_test,
        svm.predict(X_test)
    )

    knn_accuracy = accuracy_score(
        y_test,
        knn.predict(X_test)
    )

    print(f"Random Forest     : {rf_accuracy:.4f}")
    print(f"SVM               : {svm_accuracy:.4f}")
    print(f"KNN               : {knn_accuracy:.4f}")
    print(f"Voting Classifier : {voting_accuracy:.4f}")

    return (
        random_forest,
        svm,
        knn,
        voting,
        scaler,
        encoder
    )
    # =====================================================
# Save Models
# =====================================================

def save_models():

    (
        random_forest,
        svm,
        knn,
        voting,
        scaler,
        encoder
    ) = train_voting_classifier()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nSaving Models...\n")

    joblib.dump(
        random_forest,
        os.path.join(
            MODEL_DIR,
            "random_forest.pkl"
        )
    )

    joblib.dump(
        svm,
        os.path.join(
            MODEL_DIR,
            "svm.pkl"
        )
    )

    joblib.dump(
        knn,
        os.path.join(
            MODEL_DIR,
            "knn.pkl"
        )
    )

    joblib.dump(
        voting,
        os.path.join(
            MODEL_DIR,
            "voting.pkl"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            MODEL_DIR,
            "scaler.pkl"
        )
    )

    joblib.dump(
        encoder,
        os.path.join(
            MODEL_DIR,
            "label_encoder.pkl"
        )
    )

    print("=" * 60)
    print("Models Saved Successfully")
    print("=" * 60)

    print("\nSaved Files\n")

    print("✓ random_forest.pkl")
    print("✓ svm.pkl")
    print("✓ knn.pkl")
    print("✓ voting.pkl")
    print("✓ scaler.pkl")
    print("✓ label_encoder.pkl")


# =====================================================
# Main Function
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("BloodPrint AI Training Started")
    print("=" * 60)

    save_models()

    print("\n" + "=" * 60)
    print("Training Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
