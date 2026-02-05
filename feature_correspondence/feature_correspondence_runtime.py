import abc
import os
import numpy as np
from typing import List, Dict # Ensure List is imported

# Assuming matching.py and utils.py are accessible in the same environment or path
from feature_correspondence.thirdparty.sift.matching import FeatureMatcher
from feature_correspondence.thirdparty.xfeat.matching import XFeatMatcher

# Define an abstract base class for feature extraction and matching
class FeatureExtractorMatcher(abc.ABC):
    """
    Abstract base class defining the interface for feature extraction and matching.
    All concrete feature extractor/matcher implementations must inherit from this class.
    """
    @abc.abstractmethod
    def process(self, image_paths: List[str], save_plot: bool = False, dataset_name: str = "", images_color_for_plotting: List[np.ndarray] = []) -> Dict:
        """
        Abstract method to perform feature extraction and matching on a list of images.

        Args:
            image_paths: A list of file paths to the images to be processed.
            save_plot (bool): Whether to save feature/match plots.
            dataset_name (str): Name of the dataset for plot saving.
            images_color_for_plotting (List[np.ndarray]): Original color images for plotting.

        Returns:
            A dictionary containing the results of the feature extraction and matching.
            The structure of this dictionary can vary based on the specific implementation,
            but typically includes information about the method used, extracted features,
            and matched correspondences.
        """
        pass

class SIFTFeatureExtractorMatcher(FeatureExtractorMatcher):
    """
    Concrete implementation of FeatureExtractorMatcher using SIFT (Scale-Invariant Feature Transform).
    This class integrates the FeatureMatcher from matching.py.
    """
    def __init__(self):
        """
        Initializes the SIFT feature extractor and matcher by creating an instance
        of the FeatureMatcher class.
        """
        print("Initializing SIFT Feature Extractor and Matcher...")
        # Initialize the FeatureMatcher from matching.py
        self.feature_matcher = FeatureMatcher()

    def process(self, image_paths: List[str], save_plot: bool = False, dataset_name: str = "", images_color_for_plotting: List[np.ndarray] = []) -> Dict:
        """
        Performs SIFT feature extraction and matching using the FeatureMatcher class.

        Args:
            image_paths: A list of file paths to the images.
            save_plot (bool): Whether to save feature/match plots.
            dataset_name (str): Name of the dataset for plot saving.
            images_color_for_plotting (List[np.ndarray]): Original color images for plotting.

        Returns:
            A dictionary with SIFT processing results, directly from FeatureMatcher.
        """
        print(f"--- Running SIFT pipeline for {len(image_paths)} images ---")

        # Renaming for clarity, assuming these are already loaded grayscale images
        loaded_images_gray = image_paths

        # Call the actual SIFT processing pipeline from FeatureMatcher
        # The FeatureMatcher.process_images method expects grayscale images.
        results = self.feature_matcher.process_images(
            loaded_images_gray,
            save_plot=save_plot,
            dataset_name=dataset_name,
            images_color_for_plotting=images_color_for_plotting
        )
        results["method"] = "SIFT" # Add method info for consistency with factory output

        print("SIFT processing complete.")
        return results

class XFeatFeatureExtractorMatcher(FeatureExtractorMatcher):
    """
    Concrete implementation of FeatureExtractorMatcher using XFeat.
    This class simulates XFeat-based feature extraction and matching.
    """
    def __init__(self):
        """
        Initializes the XFeat feature extractor and matcher.
        In a real application, this would involve importing the XFeat library
        and initializing its model.
        """
        print("Initializing XFeat Feature Extractor and Matcher...")
        model_source = os.getenv("XFEAT_MODEL_SOURCE", "auto")
        local_repo = os.getenv("XFEAT_LOCAL_REPO")
        weights_path = os.getenv("XFEAT_WEIGHTS")
        device = os.getenv("XFEAT_DEVICE")
        input_scale = os.getenv("XFEAT_INPUT_SCALE", "auto")

        top_k = int(os.getenv("XFEAT_TOP_K", "4096"))
        detection_threshold = float(os.getenv("XFEAT_DET_THRESH", "0.05"))
        min_cossim = float(os.getenv("XFEAT_MIN_COSSIM", "0.82"))
        min_inliers = int(os.getenv("XFEAT_MIN_INLIERS", "20"))
        ransac_thresh = float(os.getenv("XFEAT_RANSAC_THRESH", "3.0"))

        self.feature_matcher = XFeatMatcher(
            top_k=top_k,
            detection_threshold=detection_threshold,
            min_cossim=min_cossim,
            min_inlier_matches=min_inliers,
            ransac_reprojection_threshold=ransac_thresh,
            model_source=model_source,
            local_repo_path=local_repo,
            weights_path=weights_path,
            device=device,
            input_scale=input_scale,
        )

    def process(self, image_paths: List[str], save_plot: bool = False, dataset_name: str = "", images_color_for_plotting: List[np.ndarray] = []) -> Dict:
        """
        Performs simulated XFeat feature extraction and matching.

        Args:
            image_paths: A list of file paths to the images.
            save_plot (bool): Whether to save feature/match plots. (Not used in simulation)
            dataset_name (str): Name of the dataset for plot saving. (Not used in simulation)
            images_color_for_plotting (List[np.ndarray]): Original color images for plotting. (Not used in simulation)

        Returns:
            A dictionary with XFeat processing results.
        """
        print(f"--- Running XFeat pipeline for {len(image_paths)} images ---")

        loaded_images_gray = image_paths

        results = self.feature_matcher.process_images(
            loaded_images_gray,
            save_plot=save_plot,
            dataset_name=dataset_name,
            images_color_for_plotting=images_color_for_plotting,
        )
        results["method"] = "XFeat"

        print("XFeat processing complete.")
        return results

class FeatureExtractorMatcherFactory:
    """
    A factory class that provides a static method to create instances of
    FeatureExtractorMatcher based on a specified method type.
    This implements the Factory Method design pattern.
    """
    @staticmethod
    def create_extractor_matcher(method_type: str) -> FeatureExtractorMatcher:
        """
        Creates and returns an appropriate FeatureExtractorMatcher instance.

        Args:
            method_type: A string indicating the desired feature extraction method
                         ('sift' or 'xfeat'). Case-insensitive.

        Returns:
            An instance of a concrete FeatureExtractorMatcher (e.g., SIFTFeatureExtractorMatcher
            or XFeatFeatureExtractorMatcher).

        Raises:
            ValueError: If an unsupported method_type is provided.
        """
        method_type_lower = method_type.lower()
        if method_type_lower == "sift":
            return SIFTFeatureExtractorMatcher()
        elif method_type_lower == "xfeat":
            return XFeatFeatureExtractorMatcher()
        else:
            raise ValueError(
                f"Unsupported feature extraction and matching method: '{method_type}'. "
                "Please choose 'sift' or 'xfeat'."
            )
