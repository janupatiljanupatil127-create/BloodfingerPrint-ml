"""
Image preprocessing utilities for fingerprint images.
"""

import cv2
import numpy as np

IMAGE_SIZE = (128, 128)


def preprocess_image(image_path):
    """
    Read and preprocess a fingerprint image.

    Steps:
    1. Read image
    2. Convert to grayscale
    3. Resize
    4. Gaussian Blur
    5. CLAHE enhancement
    6. Adaptive Threshold

    Returns:
        Preprocessed image (NumPy array)
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    gray = cv2.resize(gray, IMAGE_SIZE)

    # Remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(blurred)

    # Adaptive Threshold
    threshold = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return threshold
