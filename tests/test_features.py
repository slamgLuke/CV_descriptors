"""Tests for features.py."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from features import (
    detect_corners,
    detect_edges,
    extract_bovw_features,
    extract_fisher_vector,
    extract_hog,
    extract_lbp,
    extract_local_descriptors,
    extract_mtcd,
    train_bovw_codebook,
    train_fisher_gmm,
)


# --- detect_edges ---

def test_edges_canny_shape(gray_image):
    out = detect_edges(gray_image, "canny")
    assert out.shape == gray_image.shape


def test_edges_sobel_uint8(gray_image):
    out = detect_edges(gray_image, "sobel")
    assert out.dtype == np.uint8
    assert out.shape == gray_image.shape


def test_edges_invalid_raises(gray_image):
    with pytest.raises(ValueError, match="method"):
        detect_edges(gray_image, "laplacian")


# --- detect_corners ---

def test_corners_harris_bgr_output(gray_image):
    out = detect_corners(gray_image, "harris")
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_corners_shi_tomasi_bgr_output(gray_image):
    out = detect_corners(gray_image, "shi_tomasi")
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_corners_invalid_raises(gray_image):
    with pytest.raises(ValueError, match="method"):
        detect_corners(gray_image, "fast")


# --- extract_lbp ---

def test_lbp_is_1d(gray_image):
    hist = extract_lbp(gray_image)
    assert hist.ndim == 1


def test_lbp_normalized(gray_image):
    hist = extract_lbp(gray_image)
    assert abs(hist.sum() - 1.0) < 1e-5


def test_lbp_non_negative(gray_image):
    hist = extract_lbp(gray_image)
    assert (hist >= 0).all()


# --- extract_hog ---

def test_hog_returns_tuple(gray_image):
    result = extract_hog(gray_image)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_hog_features_float32(gray_image):
    features, _ = extract_hog(gray_image)
    assert features.dtype == np.float32
    assert features.ndim == 1


def test_hog_visualization_same_shape(gray_image):
    _, viz = extract_hog(gray_image)
    assert viz.shape == gray_image.shape
    assert viz.dtype == np.uint8


# --- extract_local_descriptors ---

def test_orb_descriptor_shape(gray_image):
    descs = extract_local_descriptors(gray_image)
    assert descs.ndim == 2
    assert descs.shape[1] == 32
    assert descs.dtype == np.float32


def test_orb_values_in_range(gray_image):
    descs = extract_local_descriptors(gray_image)
    if descs.size > 0:
        assert descs.min() >= 0.0
        assert descs.max() <= 1.0


def test_orb_empty_image_returns_empty():
    blank = np.zeros((64, 64), dtype=np.uint8)
    descs = extract_local_descriptors(blank)
    assert descs.shape == (0, 32)


# --- train_bovw_codebook / extract_bovw_features ---

@pytest.fixture
def descriptor_sets():
    rng = np.random.default_rng(0)
    return [rng.random((50, 32)).astype(np.float32) for _ in range(10)]


def test_bovw_codebook_is_kmeans(descriptor_sets):
    model = train_bovw_codebook(descriptor_sets, n_clusters=8)
    assert isinstance(model, KMeans)
    assert model.n_clusters == 8


def test_bovw_codebook_too_few_descriptors_raises():
    tiny = [np.random.rand(2, 32).astype(np.float32)]
    with pytest.raises(ValueError, match="descriptors"):
        train_bovw_codebook(tiny, n_clusters=64)


def test_bovw_features_histogram_shape(descriptor_sets):
    codebook = train_bovw_codebook(descriptor_sets, n_clusters=8)
    sample = descriptor_sets[0]
    hist = extract_bovw_features(sample, codebook)
    assert hist.shape == (8,)


def test_bovw_features_l1_normalized(descriptor_sets):
    codebook = train_bovw_codebook(descriptor_sets, n_clusters=8)
    hist = extract_bovw_features(descriptor_sets[0], codebook)
    assert abs(hist.sum() - 1.0) < 1e-5


def test_bovw_features_empty_descriptors_zeros(descriptor_sets):
    codebook = train_bovw_codebook(descriptor_sets, n_clusters=8)
    empty = np.empty((0, 32), dtype=np.float32)
    hist = extract_bovw_features(empty, codebook)
    assert (hist == 0).all()
    assert hist.shape == (8,)


# --- train_fisher_gmm / extract_fisher_vector ---

def test_fisher_gmm_is_gmm(descriptor_sets):
    model = train_fisher_gmm(descriptor_sets, n_components=4)
    assert isinstance(model, GaussianMixture)
    assert model.weights_.shape[0] == 4


def test_fisher_vector_shape(descriptor_sets):
    gmm = train_fisher_gmm(descriptor_sets, n_components=4)
    vec = extract_fisher_vector(descriptor_sets[0], gmm)
    expected_length = 4 * (1 + 2 * 32)
    assert vec.shape == (expected_length,)


def test_fisher_vector_l2_normalized(descriptor_sets):
    gmm = train_fisher_gmm(descriptor_sets, n_components=4)
    vec = extract_fisher_vector(descriptor_sets[0], gmm)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_fisher_vector_empty_descriptors_zeros(descriptor_sets):
    gmm = train_fisher_gmm(descriptor_sets, n_components=4)
    empty = np.empty((0, 32), dtype=np.float32)
    vec = extract_fisher_vector(empty, gmm)
    expected_length = 4 * (1 + 2 * 32)
    assert (vec == 0).all()
    assert vec.shape == (expected_length,)


# --- extract_mtcd ---

def test_mtcd_output_shape(leaf_mask):
    vec = extract_mtcd(leaf_mask)
    assert vec.shape == (160,)


def test_mtcd_l2_normalized(leaf_mask):
    vec = extract_mtcd(leaf_mask)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_mtcd_float32(leaf_mask):
    vec = extract_mtcd(leaf_mask)
    assert vec.dtype == np.float32


def test_mtcd_empty_mask_zeros():
    empty = np.zeros((200, 200), dtype=np.uint8)
    vec = extract_mtcd(empty)
    assert (vec == 0).all()


def test_mtcd_custom_params(leaf_mask):
    vec = extract_mtcd(leaf_mask, n_samples=64, scales=(1, 2), n_fourier=10)
    assert vec.shape == (2 * 2 * 10,)


def test_mtcd_scale_invariance(leaf_mask):
    import cv2 as _cv2
    big_mask = np.zeros((400, 400), dtype=np.uint8)
    _cv2.circle(big_mask, (200, 200), 140, 255, -1)
    v_small = extract_mtcd(leaf_mask)
    v_big = extract_mtcd(big_mask)
    # Both normalized → cosine similarity should be high for same shape
    cosine = float(np.dot(v_small, v_big))
    assert cosine > 0.8
