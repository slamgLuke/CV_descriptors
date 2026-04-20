"""Edge, corner, and descriptor extraction utilities."""

import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def detect_edges(image: np.ndarray, method: str = "canny") -> np.ndarray:
    """Detect edges using Sobel magnitude or Canny.

    Input: Grayscale image, and method ('sobel' or 'canny').
    Output: Edge map image.
    Logic: Apply selected edge detector directly on grayscale input.
    """
    if method == "canny":
        edge_map = cv2.Canny(image, 100, 200)
        return edge_map

    if method == "sobel":
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        normalized_edges = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_edges.astype(np.uint8)

    raise ValueError("method must be 'sobel' or 'canny'.")


def detect_corners(image: np.ndarray, method: str = "harris") -> np.ndarray:
    """Detect corners with Harris or Shi-Tomasi.

    Input: Grayscale image, and method ('harris' or 'shi_tomasi').
    Output: BGR image with detected corners highlighted.
    Logic: Compute corner response or points, then draw markers on visualization copy.
    """
    visualization_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if method == "harris":
        harris_response = cv2.cornerHarris(np.float32(image), 2, 3, 0.04)
        harris_response = cv2.dilate(harris_response, None)
        threshold_value = 0.01 * harris_response.max()
        visualization_image[harris_response > threshold_value] = [0, 0, 255]
        return visualization_image

    if method == "shi_tomasi":
        corners = cv2.goodFeaturesToTrack(
            image,
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

    Input: Grayscale image and LBP parameters.
    Output: Normalized LBP histogram vector.
    Logic: Compute local binary pattern image, then summarize as histogram.
    """
    lbp_image = local_binary_pattern(image, num_points, radius, method=method)

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

    Input: Grayscale image and HOG parameters.
    Output: Tuple (hog_features, hog_visualization_image).
    Logic: Run scikit-image HOG with visualization on grayscale input.
    """
    hog_features, hog_visualization = hog(
        image,
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


def extract_local_descriptors(image: np.ndarray, n_keypoints: int = 500) -> np.ndarray:
    """Extract local ORB descriptors from a grayscale image.

    Input: Grayscale image and max keypoint count.
    Output: Descriptor matrix of shape (N, 32), float32 in [0, 1].
    Logic: Detect ORB keypoints and convert binary descriptors to float vectors.
    """
    orb = cv2.ORB_create(nfeatures=n_keypoints)
    _, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        return np.empty((0, 32), dtype=np.float32)

    return (descriptors.astype(np.float32) / 255.0).astype(np.float32)


def _stack_descriptor_sets(descriptor_sets: list[np.ndarray]) -> np.ndarray:
    """Stack non-empty descriptor matrices into one 2D array."""
    non_empty_sets = [descriptors for descriptors in descriptor_sets if descriptors.size > 0]
    if not non_empty_sets:
        raise ValueError("No descriptors available to train codebook/model.")
    return np.vstack(non_empty_sets).astype(np.float32)


def train_bovw_codebook(
    descriptor_sets: list[np.ndarray],
    n_clusters: int = 64,
    random_state: int = 42,
    max_descriptors: int = 50000,
) -> KMeans:
    """Train a BoVW codebook using k-means++ initialization.

    Input: List of descriptor arrays from images.
    Output: Fitted scikit-learn KMeans model.
    Logic: Stack descriptors, optionally subsample, then fit KMeans with k-means++.
    """
    descriptor_matrix = _stack_descriptor_sets(descriptor_sets)

    if len(descriptor_matrix) > max_descriptors:
        rng = np.random.default_rng(random_state)
        selected_indices = rng.choice(len(descriptor_matrix), size=max_descriptors, replace=False)
        descriptor_matrix = descriptor_matrix[selected_indices]

    if len(descriptor_matrix) < n_clusters:
        raise ValueError(
            f"Not enough descriptors ({len(descriptor_matrix)}) for n_clusters={n_clusters}."
        )

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    )
    kmeans.fit(descriptor_matrix)
    return kmeans


def extract_bovw_features(descriptors: np.ndarray, codebook: KMeans) -> np.ndarray:
    """Extract Bag of Visual Words histogram with hard assignments.

    Input: Descriptor matrix and fitted KMeans codebook.
    Output: L1-normalized visual word histogram (float32).
    Logic: Assign each descriptor to closest centroid and count assignments.
    """
    n_clusters = int(codebook.n_clusters)

    if descriptors.size == 0:
        return np.zeros(n_clusters, dtype=np.float32)

    word_ids = codebook.predict(descriptors.astype(np.float32))
    histogram = np.bincount(word_ids, minlength=n_clusters).astype(np.float32)

    histogram_sum = histogram.sum()
    if histogram_sum > 0:
        histogram /= histogram_sum

    return histogram


def train_fisher_gmm(
    descriptor_sets: list[np.ndarray],
    n_components: int = 32,
    random_state: int = 42,
    max_descriptors: int = 50000,
) -> GaussianMixture:
    """Train a diagonal-covariance GMM for Fisher vector encoding.

    Input: List of descriptor arrays from images.
    Output: Fitted GaussianMixture model.
    Logic: Stack descriptors, optionally subsample, then fit a GMM.
    """
    descriptor_matrix = _stack_descriptor_sets(descriptor_sets)

    if len(descriptor_matrix) > max_descriptors:
        rng = np.random.default_rng(random_state)
        selected_indices = rng.choice(len(descriptor_matrix), size=max_descriptors, replace=False)
        descriptor_matrix = descriptor_matrix[selected_indices]

    if len(descriptor_matrix) < n_components:
        raise ValueError(
            f"Not enough descriptors ({len(descriptor_matrix)}) for n_components={n_components}."
        )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=200,
        reg_covar=1e-6,
        random_state=random_state,
    )
    gmm.fit(descriptor_matrix)
    return gmm


def extract_fisher_vector(descriptors: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    """Extract Fisher vector using soft assignment from a fitted GMM.

    Input: Descriptor matrix and fitted GaussianMixture model.
    Output: Power- and L2-normalized Fisher vector.
    Logic: Compute gradients wrt mixture weights, means, and variances.
    """
    n_components = int(gmm.weights_.shape[0])
    descriptor_dim = int(gmm.means_.shape[1])
    fisher_length = n_components * (1 + 2 * descriptor_dim)

    if descriptors.size == 0:
        return np.zeros(fisher_length, dtype=np.float32)

    descriptors = descriptors.astype(np.float32)
    n_descriptors = float(descriptors.shape[0])

    posterior = gmm.predict_proba(descriptors)
    weights = gmm.weights_.astype(np.float32)
    means = gmm.means_.astype(np.float32)
    variances = gmm.covariances_.astype(np.float32)
    stddev = np.sqrt(variances + 1e-12)

    s0 = posterior.sum(axis=0) / n_descriptors
    s1 = posterior.T @ descriptors / n_descriptors
    s2 = posterior.T @ (descriptors**2) / n_descriptors

    grad_weights = (s0 - weights) / np.sqrt(weights + 1e-12)
    grad_means = (s1 - s0[:, np.newaxis] * means) / (
        np.sqrt(weights + 1e-12)[:, np.newaxis] * stddev
    )
    grad_variances = (
        s2
        - 2.0 * s1 * means
        + s0[:, np.newaxis] * (means**2 - variances)
    ) / (np.sqrt(2.0 * (weights + 1e-12))[:, np.newaxis] * variances)

    fisher_vector = np.concatenate(
        [
            grad_weights.ravel(),
            grad_means.ravel(),
            grad_variances.ravel(),
        ]
    ).astype(np.float32)

    fisher_vector = np.sign(fisher_vector) * np.sqrt(np.abs(fisher_vector) + 1e-12)
    norm_value = np.linalg.norm(fisher_vector)
    if norm_value > 0:
        fisher_vector /= norm_value

    return fisher_vector.astype(np.float32)


def extract_mtcd(
    mask: np.ndarray,
    n_samples: int = 128,
    scales: tuple = (1, 4, 8, 16),
    n_fourier: int = 20,
) -> np.ndarray:
    """Extract Multiscale Triangular Centroid Distance descriptor.

    Input: Binary mask (uint8) from segment_leaf/apply_morphology, plus tuning params.
    Output: L2-normalized descriptor vector of shape (len(scales) * 2 * n_fourier,).
    Logic: At each scale, compute centroid-distance and triangle-area signatures along
           the resampled contour, then take FFT magnitudes for rotation/start invariance.
    """
    descriptor_length = len(scales) * 2 * n_fourier

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(descriptor_length, dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 1.0:
        return np.zeros(descriptor_length, dtype=np.float32)

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return np.zeros(descriptor_length, dtype=np.float32)
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 4:
        return np.zeros(descriptor_length, dtype=np.float32)

    diffs = np.diff(points, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.sqrt((diffs ** 2).sum(axis=1)))])
    total_arc = arc[-1]
    if total_arc < 1e-6:
        return np.zeros(descriptor_length, dtype=np.float32)

    sample_arc = np.linspace(0.0, total_arc, n_samples, endpoint=False)
    sampled = np.stack([
        np.interp(sample_arc, arc, points[:, 0]),
        np.interp(sample_arc, arc, points[:, 1]),
    ], axis=1)

    scale_factor = np.sqrt(area)
    all_features: list[np.ndarray] = []

    for s in scales:
        d_centroid = np.sqrt((sampled[:, 0] - cx) ** 2 + (sampled[:, 1] - cy) ** 2)
        d_centroid /= scale_factor

        idx_prev = (np.arange(n_samples) - s) % n_samples
        idx_next = (np.arange(n_samples) + s) % n_samples
        p_prev = sampled[idx_prev]
        p_next = sampled[idx_next]

        triangle_area = 0.5 * np.abs(
            p_prev[:, 0] * (sampled[:, 1] - p_next[:, 1])
            + sampled[:, 0] * (p_next[:, 1] - p_prev[:, 1])
            + p_next[:, 0] * (p_prev[:, 1] - sampled[:, 1])
        ) / (scale_factor ** 2)

        all_features.append(np.abs(np.fft.fft(d_centroid))[:n_fourier])
        all_features.append(np.abs(np.fft.fft(triangle_area))[:n_fourier])

    vector = np.concatenate(all_features).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector
