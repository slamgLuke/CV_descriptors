"""Tests for pipeline.py."""

import numpy as np
import pytest

from pipeline import LeafAnalysisPipeline


EXPECTED_RESULT_KEYS = {
    "original", "grayscale", "mask", "morphology",
    "edges", "corners", "hog_image", "lbp_features",
    "hog_features", "feature_vector",
}


# --- default init ---

def test_default_descriptor(image_file):
    pipe = LeafAnalysisPipeline()
    assert pipe.descriptor == "lbp_hog"


def test_default_apply_mask_false():
    assert LeafAnalysisPipeline().apply_mask_to_features is False


def test_default_morphology_operations_none():
    assert LeafAnalysisPipeline().morphology_operations is None


# --- run_full_extractor: keys and types ---

def test_run_full_extractor_returns_all_keys(image_file):
    pipe = LeafAnalysisPipeline()
    result = pipe.run_full_extractor(image_file)
    assert EXPECTED_RESULT_KEYS.issubset(result.keys())


def test_feature_vector_is_float32(image_file):
    pipe = LeafAnalysisPipeline()
    result = pipe.run_full_extractor(image_file)
    assert result["feature_vector"].dtype == np.float32
    assert result["feature_vector"].ndim == 1


def test_grayscale_is_2d(image_file):
    result = LeafAnalysisPipeline().run_full_extractor(image_file)
    assert result["grayscale"].ndim == 2


# --- backward compat: lbp_hog vector shape is stable ---

def test_lbp_hog_feature_vector_matches_concat(image_file):
    pipe = LeafAnalysisPipeline()
    result = pipe.run_full_extractor(image_file)
    expected = np.concatenate([result["lbp_features"], result["hog_features"]])
    np.testing.assert_array_equal(result["feature_vector"], expected)


# --- descriptor variants ---

def test_descriptor_lbp_smaller_than_lbp_hog(image_file):
    v_lbp = LeafAnalysisPipeline(descriptor="lbp").run_full_extractor(image_file)["feature_vector"]
    v_full = LeafAnalysisPipeline(descriptor="lbp_hog").run_full_extractor(image_file)["feature_vector"]
    assert len(v_lbp) < len(v_full)


def test_descriptor_hog_matches_standalone(image_file):
    from features import extract_hog
    import cv2
    result_pipe = LeafAnalysisPipeline(descriptor="hog").run_full_extractor(image_file)
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_direct, _ = extract_hog(gray)
    np.testing.assert_array_almost_equal(result_pipe["feature_vector"], hog_direct)


def test_descriptor_combination_hog_plus_lbp(image_file):
    v_combo = LeafAnalysisPipeline(descriptor="hog+lbp").run_full_extractor(image_file)["feature_vector"]
    v_hog = LeafAnalysisPipeline(descriptor="hog").run_full_extractor(image_file)["feature_vector"]
    v_lbp = LeafAnalysisPipeline(descriptor="lbp").run_full_extractor(image_file)["feature_vector"]
    assert len(v_combo) == len(v_hog) + len(v_lbp)


def test_descriptor_invalid_raises(image_file):
    pipe = LeafAnalysisPipeline(descriptor="unknown_desc")
    with pytest.raises(ValueError, match="Unknown descriptor"):
        pipe.run_full_extractor(image_file)


# --- mtcd ---

def test_descriptor_mtcd_requires_segmentation(image_file):
    pipe = LeafAnalysisPipeline(descriptor="mtcd")
    with pytest.raises(ValueError, match="segmentation_method"):
        pipe.run_full_extractor(image_file)


def test_descriptor_mtcd_with_segmentation(image_file):
    pipe = LeafAnalysisPipeline(descriptor="mtcd", segmentation_method="otsu")
    result = pipe.run_full_extractor(image_file)
    assert result["feature_vector"].shape == (160,)


# --- apply_mask_to_features ---

def test_apply_mask_changes_feature_vector(image_file):
    v_no_mask = LeafAnalysisPipeline(segmentation_method="otsu").run_full_extractor(image_file)["feature_vector"]
    v_masked = LeafAnalysisPipeline(
        segmentation_method="otsu", apply_mask_to_features=True
    ).run_full_extractor(image_file)["feature_vector"]
    assert not np.array_equal(v_no_mask, v_masked)


# --- morphology_operations (sequential) ---

def test_morphology_operations_runs(image_file):
    pipe = LeafAnalysisPipeline(
        segmentation_method="otsu",
        morphology_operations=["opening", "closing"],
    )
    result = pipe.run_full_extractor(image_file)
    assert result["morphology"] is not None


def test_morphology_operations_precedence_over_single(image_file):
    # morphology_operations takes precedence: single erosion vs triple erosion must differ
    pipe_single = LeafAnalysisPipeline(
        segmentation_method="otsu",
        morphology_operation="erosion",
    )
    pipe_list = LeafAnalysisPipeline(
        segmentation_method="otsu",
        morphology_operation="erosion",           # ignored — overridden by the list
        morphology_operations=["erosion", "erosion", "erosion"],
    )
    r_single = pipe_single.run_full_extractor(image_file)["morphology"]
    r_list = pipe_list.run_full_extractor(image_file)["morphology"]
    assert not np.array_equal(r_single, r_list)


# --- fit / transform ---

def test_fit_is_noop_for_lbp_hog(many_image_files):
    pipe = LeafAnalysisPipeline()
    returned = pipe.fit(many_image_files)
    assert returned is pipe
    assert pipe._codebook is None
    assert pipe._gmm is None


def test_fit_trains_codebook_for_bovw(many_image_files):
    pipe = LeafAnalysisPipeline(descriptor="bovw")
    pipe.fit(many_image_files)
    assert pipe._codebook is not None


def test_fit_trains_gmm_for_fisher(many_image_files):
    pipe = LeafAnalysisPipeline(descriptor="fisher")
    pipe.fit(many_image_files)
    assert pipe._gmm is not None


def test_bovw_without_fit_raises(image_file):
    pipe = LeafAnalysisPipeline(descriptor="bovw")
    with pytest.raises(RuntimeError, match="fit"):
        pipe.run_full_extractor(image_file)


def test_transform_returns_1d_array(image_file):
    pipe = LeafAnalysisPipeline()
    vec = pipe.transform(image_file)
    assert vec.ndim == 1
    assert vec.dtype == np.float32


def test_transform_matches_run_full_extractor(image_file):
    pipe = LeafAnalysisPipeline()
    vec_transform = pipe.transform(image_file)
    vec_full = pipe.run_full_extractor(image_file)["feature_vector"]
    np.testing.assert_array_equal(vec_transform, vec_full)


# --- error cases ---

def test_file_not_found_raises():
    pipe = LeafAnalysisPipeline()
    with pytest.raises(FileNotFoundError):
        pipe.run_full_extractor("/nonexistent/path/image.jpg")
