"""Tests for dataset.py."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset import build_dataset, list_class_dirs, list_images


# ---------------------------------------------------------------------------
# Helpers to build a fake PlantVillage-style directory
# ---------------------------------------------------------------------------

def _make_class_dir(root: Path, class_name: str, n_images: int, seed: int = 0) -> Path:
    cls_dir = root / class_name
    cls_dir.mkdir()
    rng = np.random.default_rng(seed)
    for i in range(n_images):
        img = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(cls_dir / f"img_{i:03d}.jpg"), img)
    return cls_dir


@pytest.fixture
def plant_root(tmp_path):
    """Fake PlantVillage with 3 classes."""
    _make_class_dir(tmp_path, "Tomato_healthy", 10, seed=0)
    _make_class_dir(tmp_path, "Tomato_Early_blight", 15, seed=1)
    _make_class_dir(tmp_path, "Tomato_Late_blight", 12, seed=2)
    return tmp_path


# --- list_class_dirs ---

def test_list_class_dirs_returns_all(plant_root):
    dirs = list_class_dirs(plant_root)
    assert len(dirs) == 3


def test_list_class_dirs_returns_paths(plant_root):
    dirs = list_class_dirs(plant_root)
    assert all(isinstance(d, Path) for d in dirs)


def test_list_class_dirs_sorted(plant_root):
    dirs = list_class_dirs(plant_root)
    names = [d.name for d in dirs]
    assert names == sorted(names)


def test_list_class_dirs_filter(plant_root):
    dirs = list_class_dirs(plant_root, filter_substring="blight")
    assert len(dirs) == 2
    assert all("blight" in d.name.lower() for d in dirs)


def test_list_class_dirs_filter_no_match(plant_root):
    dirs = list_class_dirs(plant_root, filter_substring="pepper")
    assert dirs == []


# --- list_images ---

def test_list_images_count(plant_root):
    cls_dir = plant_root / "Tomato_healthy"
    imgs = list_images(cls_dir)
    assert len(imgs) == 10


def test_list_images_returns_paths(plant_root):
    imgs = list_images(plant_root / "Tomato_healthy")
    assert all(isinstance(p, Path) for p in imgs)


def test_list_images_sorted(plant_root):
    imgs = list_images(plant_root / "Tomato_healthy")
    assert imgs == sorted(imgs)


def test_list_images_extension_filter(tmp_path):
    cls_dir = tmp_path / "cls"
    cls_dir.mkdir()
    (cls_dir / "a.jpg").write_bytes(b"")
    (cls_dir / "b.txt").write_bytes(b"")
    imgs = list_images(cls_dir)
    assert len(imgs) == 1
    assert imgs[0].suffix == ".jpg"


# --- build_dataset: binary ---

def test_binary_label_values(plant_root):
    _, labels, label_names = build_dataset(plant_root, mode="binary", balance=False)
    assert set(labels).issubset({0, 1})
    assert label_names == ["healthy", "not_healthy"]


def test_binary_healthy_label_zero(plant_root):
    paths, labels, _ = build_dataset(plant_root, mode="binary", balance=False)
    for p, lb in zip(paths, labels):
        if "healthy" in Path(p).parent.name.lower():
            assert lb == 0


def test_binary_balanced_equal_counts(plant_root):
    paths, labels, _ = build_dataset(plant_root, mode="binary", balance=True)
    n_healthy = (labels == 0).sum()
    n_sick = (labels == 1).sum()
    assert n_healthy == n_sick


# --- build_dataset: multiclass ---

def test_multiclass_n_classes(plant_root):
    _, labels, label_names = build_dataset(plant_root, mode="multiclass", balance=False)
    assert len(label_names) == 3
    assert set(labels) == {0, 1, 2}


def test_multiclass_balanced(plant_root):
    _, labels, _ = build_dataset(plant_root, mode="multiclass", balance=True)
    counts = [int((labels == i).sum()) for i in range(3)]
    assert len(set(counts)) == 1


# --- build_dataset: sample_fraction ---

def test_sample_fraction_reduces_dataset(plant_root):
    _, labels_full, _ = build_dataset(plant_root, mode="multiclass", balance=False, sample_fraction=1.0)
    _, labels_half, _ = build_dataset(plant_root, mode="multiclass", balance=False, sample_fraction=0.5)
    assert len(labels_half) < len(labels_full)


# --- build_dataset: filter_substring ---

def test_filter_substring_reduces_classes(plant_root):
    _, labels, label_names = build_dataset(
        plant_root, mode="multiclass", filter_substring="blight", balance=False
    )
    assert len(label_names) == 2


# --- build_dataset: error cases ---

def test_empty_root_raises(tmp_path):
    with pytest.raises(ValueError, match="No class directories"):
        build_dataset(tmp_path, mode="binary")


def test_invalid_mode_raises(plant_root):
    with pytest.raises(ValueError, match="mode"):
        build_dataset(plant_root, mode="triplet")
