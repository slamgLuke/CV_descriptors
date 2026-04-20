"""Tests for classifiers.py."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from classifiers import train_and_evaluate


@pytest.fixture
def binary_data():
    """60 samples, 2 linearly separable classes, 5 features."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (30, 5)), rng.normal(3, 0.3, (30, 5))])
    y = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    return X.astype(np.float32), y


@pytest.fixture
def multiclass_data():
    """90 samples, 3 well-separated classes, 4 features."""
    rng = np.random.default_rng(1)
    centres = [(0, 0, 0, 0), (5, 5, 0, 0), (0, 0, 5, 5)]
    X = np.vstack([rng.normal(c, 0.3, (30, 4)) for c in centres])
    y = np.array([i for i in range(3) for _ in range(30)], dtype=np.int64)
    return X.astype(np.float32), y


# --- return structure ---

def test_returns_all_models(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y)
    assert set(results.keys()) == {"kNN", "SVM", "RandomForest"}


def test_each_model_has_required_keys(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y)
    required = {"accuracy", "f1_score", "confusion_matrix", "confusion_matrix_figure"}
    for model_name, metrics in results.items():
        assert required.issubset(metrics.keys()), f"{model_name} missing keys"


def test_confusion_matrix_figure_is_matplotlib(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y)
    for metrics in results.values():
        assert isinstance(metrics["confusion_matrix_figure"], Figure)


# --- metric values ---

def test_accuracy_in_valid_range(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y)
    for metrics in results.values():
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_f1_in_valid_range(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y)
    for metrics in results.values():
        assert 0.0 <= metrics["f1_score"] <= 1.0


def test_separable_data_high_accuracy(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y)
    assert results["SVM"]["accuracy"] > 0.9


def test_multiclass_confusion_matrix_shape(multiclass_data):
    X, y = multiclass_data
    results = train_and_evaluate(X, y)
    for metrics in results.values():
        assert metrics["confusion_matrix"].shape == (3, 3)


# --- scale parameter ---

def test_scale_false_runs_without_error(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y, scale=False)
    assert set(results.keys()) == {"kNN", "SVM", "RandomForest"}


def test_scale_true_same_keys(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y, scale=True)
    assert set(results.keys()) == {"kNN", "SVM", "RandomForest"}


# --- tune parameter ---

def test_tune_runs_without_error(binary_data):
    X, y = binary_data
    results = train_and_evaluate(X, y, tune=True)
    assert set(results.keys()) == {"kNN", "SVM", "RandomForest"}


# --- input validation ---

def test_raises_on_none_X(binary_data):
    _, y = binary_data
    with pytest.raises(ValueError):
        train_and_evaluate(None, y)


def test_raises_on_empty_X():
    with pytest.raises(ValueError):
        train_and_evaluate(np.array([]), np.array([]))


def test_raises_on_length_mismatch():
    X = np.ones((10, 3), dtype=np.float32)
    y = np.zeros(5, dtype=np.int64)
    with pytest.raises(ValueError):
        train_and_evaluate(X, y)
