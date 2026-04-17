"""Edge, corner, and descriptor extraction utilities."""

import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale when needed.

    Input: BGR or grayscale NumPy image.
    Output: Grayscale NumPy image.
    Logic: Detect channel count and convert with OpenCV if image is color.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 2:
        return image.copy()

    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError("Unsupported image format.")


def detect_edges(image: np.ndarray, method: str = "canny") -> np.ndarray:
    """Detect edges using Sobel magnitude or Canny.

    Input: BGR or grayscale image, and method ('sobel' or 'canny').
    Output: Edge map image.
    Logic: Convert to grayscale and apply selected edge detector.
    """
    grayscale_image = _ensure_grayscale(image)

    if method == "canny":
        edge_map = cv2.Canny(grayscale_image, 100, 200)
        return edge_map

    if method == "sobel":
        sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        normalized_edges = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_edges.astype(np.uint8)

    raise ValueError("method must be 'sobel' or 'canny'.")


def detect_corners(image: np.ndarray, method: str = "harris") -> np.ndarray:
    """Detect corners with Harris or Shi-Tomasi.

    Input: BGR or grayscale image, and method ('harris' or 'shi_tomasi').
    Output: BGR image with detected corners highlighted.
    Logic: Compute corner response or points, then draw markers on image copy.
    """
    grayscale_image = _ensure_grayscale(image)
    visualization_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    if method == "harris":
        harris_response = cv2.cornerHarris(np.float32(grayscale_image), 2, 3, 0.04)
        harris_response = cv2.dilate(harris_response, None)
        threshold_value = 0.01 * harris_response.max()
        visualization_image[harris_response > threshold_value] = [0, 0, 255]
        return visualization_image

    if method == "shi_tomasi":
        corners = cv2.goodFeaturesToTrack(
            grayscale_image,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
        )

        if corners is None:
            return visualization_image

        corners = np.int32(corners)
        for corner in corners:
            x_coordinate = int(corner.ravel()[0])
            y_coordinate = int(corner.ravel()[1])
            cv2.circle(visualization_image, (x_coordinate, y_coordinate), 3, (0, 255, 0), -1)
        return visualization_image

    raise ValueError("method must be 'harris' or 'shi_tomasi'.")


def extract_lbp(
    image: np.ndarray,
    radius: int = 1,
    num_points: int = 8,
    method: str = "uniform",
) -> np.ndarray:
    """Extract an LBP histogram descriptor.

    Input: BGR or grayscale image and LBP parameters.
    Output: Normalized LBP histogram vector.
    Logic: Compute local binary pattern image, then summarize as histogram.
    """
    grayscale_image = _ensure_grayscale(image)
    lbp_image = local_binary_pattern(grayscale_image, num_points, radius, method=method)

    num_bins = int(lbp_image.max() + 1)
    histogram, _ = np.histogram(lbp_image.ravel(), bins=num_bins, range=(0, num_bins))
    histogram = histogram.astype(np.float32)
    histogram_sum = histogram.sum()

    if histogram_sum > 0:
        histogram = histogram / histogram_sum

    return histogram


def extract_hog(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple = (8, 8),
    cells_per_block: tuple = (2, 2),
) -> tuple:
    """Extract HOG feature vector and visualization image.

    Input: BGR or grayscale image and HOG parameters.
    Output: Tuple (hog_features, hog_visualization_image).
    Logic: Convert to grayscale and run scikit-image HOG with visualization.
    """
    grayscale_image = _ensure_grayscale(image)

    hog_features, hog_visualization = hog(
        grayscale_image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )

    hog_visualization = cv2.normalize(hog_visualization, None, 0, 255, cv2.NORM_MINMAX)
    hog_visualization = hog_visualization.astype(np.uint8)
    return hog_features.astype(np.float32), hog_visualization


# Placeholder: custom descriptor and Bag of Words will be added later.
