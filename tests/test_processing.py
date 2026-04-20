"""Tests for processing.py."""

import numpy as np
import pytest

from processing import apply_morphology, apply_preprocessing, segment_leaf


# --- apply_preprocessing ---

def test_gaussian_preserves_shape(gray_image):
    out = apply_preprocessing(gray_image, "gaussian")
    assert out.shape == gray_image.shape
    assert out.dtype == np.uint8


def test_median_preserves_shape(gray_image):
    out = apply_preprocessing(gray_image, "median")
    assert out.shape == gray_image.shape
    assert out.dtype == np.uint8


def test_gaussian_actually_blurs(gray_image):
    out = apply_preprocessing(gray_image, "gaussian")
    assert float(out.std()) < float(gray_image.std())


def test_preprocessing_invalid_method_raises(gray_image):
    with pytest.raises(ValueError, match="filter_type"):
        apply_preprocessing(gray_image, "sharpen")


# --- segment_leaf ---

def test_segment_otsu_binary_values(gray_image):
    mask = segment_leaf(gray_image, "otsu")
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask.shape == gray_image.shape


def test_segment_global_binary_values(gray_image):
    mask = segment_leaf(gray_image, "global")
    assert set(np.unique(mask)).issubset({0, 255})


def test_segment_invalid_method_raises(gray_image):
    with pytest.raises(ValueError, match="method"):
        segment_leaf(gray_image, "watershed")


# --- apply_morphology ---

@pytest.mark.parametrize("op", ["erosion", "dilation", "opening", "closing"])
def test_morphology_preserves_shape(leaf_mask, op):
    out = apply_morphology(leaf_mask, op)
    assert out.shape == leaf_mask.shape
    assert out.dtype == np.uint8


def test_erosion_reduces_foreground(leaf_mask):
    out = apply_morphology(leaf_mask, "erosion")
    assert out.sum() <= leaf_mask.sum()


def test_dilation_increases_foreground(leaf_mask):
    out = apply_morphology(leaf_mask, "dilation")
    assert out.sum() >= leaf_mask.sum()


def test_morphology_none_mask_raises():
    with pytest.raises(ValueError, match="None"):
        apply_morphology(None, "erosion")


def test_morphology_invalid_op_raises(leaf_mask):
    with pytest.raises(ValueError, match="operation"):
        apply_morphology(leaf_mask, "gradient")
