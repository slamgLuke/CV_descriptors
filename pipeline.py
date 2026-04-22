"""Pipeline manager for the full leaf analysis workflow."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import cv2
import numpy as np

from features import detect_corners
from features import detect_edges
from features import extract_bovw_features
from features import extract_fisher_vector
from features import extract_hog
from features import extract_hsv_lbp
from features import extract_lbp
from features import extract_local_descriptors
from features import train_bovw_codebook
from features import train_fisher_gmm
from processing import apply_morphology
from processing import apply_preprocessing
from processing import segment_leaf


class LeafAnalysisPipeline:
    """Orchestrates preprocessing, segmentation, morphology, and feature extraction."""

    def __init__(
        self,
        preprocess_filter: Optional[str] = None,
        segmentation_method: Optional[str] = None,
        morphology_operation: Optional[str] = None,
        edge_method: Optional[str] = None,
        corner_method: Optional[str] = None,
        descriptor: str = "lbp_hog",
        apply_mask_to_features: bool = False,
        morphology_operations: Optional[List[str]] = None,
    ) -> None:
        """Initialize pipeline configuration.

        descriptor: which feature vector to produce — 'lbp_hog' (default), 'lbp', 'hog',
                    'bovw', 'fisher', 'hsv_lbp', or '+'-joined combos like 'hog+hsv_lbp'.
        apply_mask_to_features: if True, zero-out background before extracting features.
        morphology_operations: ordered list of operations (e.g. ['opening', 'closing']).
                               Takes precedence over morphology_operation when set.
        """
        self.preprocess_filter = preprocess_filter
        self.segmentation_method = segmentation_method
        self.morphology_operation = morphology_operation
        self.edge_method = edge_method
        self.corner_method = corner_method
        self.descriptor = descriptor
        self.apply_mask_to_features = apply_mask_to_features
        self.morphology_operations = morphology_operations
        self._codebook = None
        self._gmm = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_image(
        self, image_path: str
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Load image, apply preprocessing/segmentation/morphology.

        Returns (grayscale_image, working_mask, feature_image, feature_image_bgr).
        feature_image is grayscale with background zeroed when apply_mask_to_features=True.
        feature_image_bgr is the BGR equivalent (needed for color-aware descriptors like hsv_lbp).
        """
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if self.preprocess_filter is not None:
            gray = apply_preprocessing(gray, filter_type=self.preprocess_filter)

        if gray is None or gray.ndim != 2:
            raise ValueError("Pipeline expects a valid grayscale image (2D array).")

        leaf_mask = None
        if self.segmentation_method is not None:
            leaf_mask = segment_leaf(gray, method=self.segmentation_method)

        working_mask = leaf_mask
        if self.morphology_operations is not None and leaf_mask is not None:
            result = leaf_mask
            for op in self.morphology_operations:
                result = apply_morphology(result, operation=op)
            working_mask = result
        elif self.morphology_operation is not None and leaf_mask is not None:
            working_mask = apply_morphology(leaf_mask, operation=self.morphology_operation)

        feature_image = gray.copy()
        if self.apply_mask_to_features and working_mask is not None:
            feature_image[working_mask == 0] = 0

        feature_image_bgr = original.copy()
        if self.preprocess_filter is not None:
            feature_image_bgr = apply_preprocessing(feature_image_bgr, filter_type=self.preprocess_filter)
        if self.apply_mask_to_features and working_mask is not None:
            feature_image_bgr[working_mask == 0] = 0

        return gray, working_mask, feature_image, feature_image_bgr

    def _compute_feature_vector(
        self,
        feature_image: np.ndarray,
        working_mask: Optional[np.ndarray],
        feature_image_bgr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build the feature vector according to self.descriptor."""
        parts = [d.strip() for d in self.descriptor.split("+")]
        vectors: list[np.ndarray] = []

        for part in parts:
            if part == "lbp_hog":
                lbp = extract_lbp(feature_image)
                hog_feats, _ = extract_hog(feature_image)
                vectors.append(np.concatenate([lbp, hog_feats]))
            elif part == "lbp":
                vectors.append(extract_lbp(feature_image))
            elif part == "hog":
                hog_feats, _ = extract_hog(feature_image)
                vectors.append(hog_feats)
            elif part == "bovw":
                if self._codebook is None:
                    raise RuntimeError("Call fit() before using descriptor='bovw'.")
                descs = extract_local_descriptors(feature_image)
                vectors.append(extract_bovw_features(descs, self._codebook))
            elif part == "fisher":
                if self._gmm is None:
                    raise RuntimeError("Call fit() before using descriptor='fisher'.")
                descs = extract_local_descriptors(feature_image)
                vectors.append(extract_fisher_vector(descs, self._gmm))
            elif part == "hsv_lbp":
                if feature_image_bgr is None:
                    raise RuntimeError("feature_image_bgr is required for descriptor='hsv_lbp'.")
                vectors.append(extract_hsv_lbp(feature_image_bgr))
            else:
                raise ValueError(
                    f"Unknown descriptor '{part}'. "
                    "Valid values: lbp, hog, lbp_hog, bovw, fisher, hsv_lbp."
                )

        return np.concatenate(vectors).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, image_paths: List[str]) -> "LeafAnalysisPipeline":
        """Train the encoding model for descriptors that require it (bovw, fisher).

        No-op for lbp, hog, lbp_hog, and hsv_lbp descriptors.
        """
        parts = {d.strip() for d in self.descriptor.split("+")}
        needs_fit = {"bovw", "fisher"} & parts
        if not needs_fit:
            return self

        descriptor_sets: list[np.ndarray] = []
        for path in image_paths:
            try:
                _, _, feature_image, _ = self._preprocess_image(path)
                descs = extract_local_descriptors(feature_image)
                descriptor_sets.append(descs)
            except (FileNotFoundError, ValueError):
                continue

        if "bovw" in needs_fit:
            self._codebook = train_bovw_codebook(descriptor_sets)
        if "fisher" in needs_fit:
            self._gmm = train_fisher_gmm(descriptor_sets)

        return self

    def transform(self, image_path: str) -> np.ndarray:
        """Return only the feature vector for an image (no intermediate results dict)."""
        _, working_mask, feature_image, feature_image_bgr = self._preprocess_image(image_path)
        return self._compute_feature_vector(feature_image, working_mask, feature_image_bgr)

    def run_full_extractor(self, image_path: str) -> Dict[str, Any]:
        """Run the full pipeline on a single image path.

        Returns a dictionary with all intermediate images and the final feature vector.
        Keys: original, grayscale, mask, morphology, edges, corners,
              hog_image, lbp_features, hog_features, feature_vector.
        """
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if self.preprocess_filter is not None:
            gray = apply_preprocessing(gray, filter_type=self.preprocess_filter)

        if gray is None or gray.ndim != 2:
            raise ValueError("Pipeline expects a valid grayscale image (2D array).")

        leaf_mask = None
        if self.segmentation_method is not None:
            leaf_mask = segment_leaf(gray, method=self.segmentation_method)

        working_mask = leaf_mask
        morphology_result = None
        if self.morphology_operations is not None and leaf_mask is not None:
            result = leaf_mask
            for op in self.morphology_operations:
                result = apply_morphology(result, operation=op)
            morphology_result = result
            working_mask = morphology_result
        elif self.morphology_operation is not None and leaf_mask is not None:
            morphology_result = apply_morphology(leaf_mask, operation=self.morphology_operation)
            working_mask = morphology_result

        edge_result = None
        if self.edge_method is not None:
            edge_result = detect_edges(gray, method=self.edge_method)

        corner_result = None
        if self.corner_method is not None:
            corner_result = detect_corners(gray, method=self.corner_method)

        feature_image = gray.copy()
        if self.apply_mask_to_features and working_mask is not None:
            feature_image[working_mask == 0] = 0

        feature_image_bgr = original.copy()
        if self.preprocess_filter is not None:
            feature_image_bgr = apply_preprocessing(feature_image_bgr, filter_type=self.preprocess_filter)
        if self.apply_mask_to_features and working_mask is not None:
            feature_image_bgr[working_mask == 0] = 0

        lbp_features = extract_lbp(feature_image)
        hog_features, hog_visualization = extract_hog(feature_image)
        final_feature_vector = self._compute_feature_vector(feature_image, working_mask, feature_image_bgr)

        return {
            "original": original,
            "grayscale": gray,
            "mask": leaf_mask,
            "morphology": morphology_result,
            "edges": edge_result,
            "corners": corner_result,
            "hog_image": hog_visualization,
            "lbp_features": lbp_features,
            "hog_features": hog_features,
            "feature_vector": final_feature_vector,
        }
