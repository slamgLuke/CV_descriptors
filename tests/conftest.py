"""Shared fixtures for all test modules."""

import cv2
import numpy as np
import pytest


@pytest.fixture
def gray_image() -> np.ndarray:
    """128×128 grayscale image with natural-looking texture."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


@pytest.fixture
def leaf_image() -> np.ndarray:
    """200×200 synthetic leaf: bright ellipse on dark background with texture."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(img, (100, 100), (75, 55), 0, 0, 360, 180, -1)
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 40, img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def leaf_mask() -> np.ndarray:
    """200×200 binary mask with a filled circle."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 70, 255, -1)
    return mask


@pytest.fixture
def image_file(tmp_path, leaf_image) -> str:
    """Save leaf_image as a BGR JPEG and return its path."""
    bgr = cv2.cvtColor(leaf_image, cv2.COLOR_GRAY2BGR)
    path = tmp_path / "leaf.png"
    cv2.imwrite(str(path), bgr)
    return str(path)


@pytest.fixture
def many_image_files(tmp_path) -> list[str]:
    """20 synthetic images varied enough for ORB to find keypoints."""
    paths = []
    for i in range(20):
        rng = np.random.default_rng(i)
        img = rng.integers(30, 226, (200, 200), dtype=np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        p = tmp_path / f"img_{i:02d}.jpg"
        cv2.imwrite(str(p), bgr)
        paths.append(str(p))
    return paths
