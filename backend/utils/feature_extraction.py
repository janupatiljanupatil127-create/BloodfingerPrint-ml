"""
Feature extraction for fingerprint images using
HOG (Histogram of Oriented Gradients)
and LBP (Local Binary Pattern).
"""

import numpy as np
from skimage.feature import hog, local_binary_pattern

# LBP Parameters
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = "uniform"


def extract_hog(image):
    """
    Extract HOG features.
    """

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )

    return features


def extract_lbp(image):
    """
    Extract LBP histogram features.
    """

    lbp = local_binary_pattern(
        image,
        N_POINTS,
        RADIUS,
        METHOD
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-7)

    return hist


def extract_features(image):
    """
    Combine HOG and LBP features.
    """

    hog_features = extract_hog(image)

    lbp_features = extract_lbp(image)

    return np.concatenate([hog_features, lbp_features])
