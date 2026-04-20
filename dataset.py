"""Dataset loading utilities for PlantVillage-style directory trees."""

from pathlib import Path
from typing import Optional

import numpy as np


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_class_dirs(
    root: Path,
    filter_substring: Optional[str] = None,
) -> list[Path]:
    """Return sorted subdirectories of root, optionally filtered by name substring."""
    root = Path(root)
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if filter_substring is not None:
        dirs = [d for d in dirs if filter_substring.lower() in d.name.lower()]
    return dirs


def list_images(
    class_dir: Path,
    exts: tuple[str, ...] = tuple(_IMAGE_EXTENSIONS),
) -> list[Path]:
    """Return all image paths inside a class directory."""
    return sorted([f for f in Path(class_dir).iterdir() if f.suffix.lower() in exts])


def build_dataset(
    root: Path,
    mode: str = "binary",
    filter_substring: Optional[str] = None,
    sample_fraction: float = 1.0,
    balance: bool = True,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Build a flat (paths, labels, label_names) dataset from a PlantVillage directory.

    mode='binary'     — labels 0 (healthy) vs 1 (not_healthy), detected by 'healthy' in dir name.
    mode='multiclass' — one integer label per class directory.
    Returns (image_paths, labels, label_names).
    """
    rng = np.random.default_rng(random_state)
    class_dirs = list_class_dirs(root, filter_substring=filter_substring)

    if not class_dirs:
        raise ValueError(f"No class directories found in {root!r}.")

    if mode == "binary":
        label_names = ["healthy", "not_healthy"]
        groups: dict[int, list[str]] = {0: [], 1: []}
        for class_dir in class_dirs:
            label = 0 if "healthy" in class_dir.name.lower() else 1
            groups[label].extend(str(p) for p in list_images(class_dir))
        paths_per_class = [groups[0], groups[1]]
        labels_per_class = [[0] * len(groups[0]), [1] * len(groups[1])]

    elif mode == "multiclass":
        label_names = [d.name for d in class_dirs]
        paths_per_class = [[str(p) for p in list_images(d)] for d in class_dirs]
        labels_per_class = [[i] * len(paths_per_class[i]) for i in range(len(class_dirs))]

    else:
        raise ValueError("mode must be 'binary' or 'multiclass'.")

    sampled_paths: list[list[str]] = []
    sampled_labels: list[list[int]] = []
    for paths, labels in zip(paths_per_class, labels_per_class):
        n = max(1, int(len(paths) * sample_fraction))
        n = min(n, len(paths))
        idx = rng.choice(len(paths), size=n, replace=False)
        sampled_paths.append([paths[i] for i in idx])
        sampled_labels.append([labels[i] for i in idx])

    if balance:
        min_count = min(len(p) for p in sampled_paths)
        balanced: list[tuple[list[str], list[int]]] = []
        for paths, labels in zip(sampled_paths, sampled_labels):
            idx = rng.choice(len(paths), size=min_count, replace=False)
            balanced.append(([paths[i] for i in idx], [labels[i] for i in idx]))
        sampled_paths = [b[0] for b in balanced]
        sampled_labels = [b[1] for b in balanced]

    flat_paths: list[str] = [p for group in sampled_paths for p in group]
    flat_labels: list[int] = [lb for group in sampled_labels for lb in group]

    return flat_paths, np.array(flat_labels, dtype=np.int64), label_names
