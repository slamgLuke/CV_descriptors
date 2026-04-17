"""Image preparation, segmentation, and morphology utilities."""

import cv2
import numpy as np


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


def apply_preprocessing(image: np.ndarray, filter_type: str = "gaussian") -> np.ndarray:
    """Convert to grayscale and apply blur filtering.

    Input: BGR or grayscale image, and filter type ('gaussian' or 'median').
    Output: Preprocessed grayscale image.
    Logic: Convert to grayscale, then apply selected blur to reduce noise.
    """
    grayscale_image = _ensure_grayscale(image)

    if filter_type == "gaussian":
        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        return blurred_image

    if filter_type == "median":
        blurred_image = cv2.medianBlur(grayscale_image, 5)
        return blurred_image

    raise ValueError("filter_type must be 'gaussian' or 'median'.")


def segment_leaf(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Segment leaf region with thresholding.

    Input: BGR or grayscale image, and method ('global' or 'otsu').
    Output: Binary mask image (uint8: 0 or 255).
    Logic: Convert to grayscale and apply thresholding to isolate foreground.
    """
    grayscale_image = _ensure_grayscale(image)

    if method == "global":
        _, binary_mask = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
        return binary_mask

    if method == "otsu":
        _, binary_mask = cv2.threshold(
            grayscale_image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary_mask

    raise ValueError("method must be 'global' or 'otsu'.")


def apply_morphology(
    mask: np.ndarray,
    operation: str = "closing",
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological operation to a binary mask.

    Input: Binary mask, operation name, kernel size, and iteration count.
    Output: Morphologically processed binary mask.
    Logic: Build structuring element and run selected morphology operation.
    """
    if mask is None:
        raise ValueError("Input mask is None.")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == "erosion":
        return cv2.erode(mask, kernel, iterations=iterations)

    if operation == "dilation":
        return cv2.dilate(mask, kernel, iterations=iterations)

    if operation == "opening":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    if operation == "closing":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    raise ValueError("operation must be 'erosion', 'dilation', 'opening', or 'closing'.")
