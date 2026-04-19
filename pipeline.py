"""Pipeline manager for the full leaf analysis workflow."""

from typing import Any
from typing import Dict
from typing import Optional

import cv2
import numpy as np

from features import detect_corners
from features import detect_edges
from features import extract_hog
from features import extract_lbp
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
    ) -> None:
        """Initialize pipeline configuration.

        Input: Optional method names for each pipeline stage.
        Output: Configured pipeline instance.
        Logic: Store selected methods where None means skip that step.
        """
        self.preprocess_filter = preprocess_filter
        self.segmentation_method = segmentation_method
        self.morphology_operation = morphology_operation
        self.edge_method = edge_method
        self.corner_method = corner_method

    def run_full_extractor(self, image_path: str) -> Dict[str, Any]:
        """Run the full pipeline on a single image path.

        Input: Image file path.
        Output: Dictionary with intermediate images and final feature vector.
        Logic: Execute selected stages and skip stages configured as None.
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")

        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        if self.preprocess_filter is not None:
            grayscale_image = apply_preprocessing(grayscale_image, filter_type=self.preprocess_filter)

        if grayscale_image is None or grayscale_image.ndim != 2:
            raise ValueError("Pipeline expects a valid grayscale image (2D array).")

        leaf_mask = None
        if self.segmentation_method is not None:
            leaf_mask = segment_leaf(grayscale_image, method=self.segmentation_method)

        morphology_result = None
        if self.morphology_operation is not None and leaf_mask is not None:
            morphology_result = apply_morphology(leaf_mask, operation=self.morphology_operation)

        edge_result = None
        if self.edge_method is not None:
            edge_result = detect_edges(grayscale_image, method=self.edge_method)

        corner_result = None
        if self.corner_method is not None:
            corner_result = detect_corners(grayscale_image, method=self.corner_method)

        lbp_features = extract_lbp(grayscale_image)
        hog_features, hog_visualization = extract_hog(grayscale_image)
        final_feature_vector = np.concatenate([lbp_features, hog_features]).astype(np.float32)

        results = {
            "original": original_image,
            "grayscale": grayscale_image,
            "mask": leaf_mask,
            "morphology": morphology_result,
            "edges": edge_result,
            "corners": corner_result,
            "hog_image": hog_visualization,
            "lbp_features": lbp_features,
            "hog_features": hog_features,
            "feature_vector": final_feature_vector,
        }
        return results
