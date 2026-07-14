import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

IMAGE_SIZE = (128, 128)
