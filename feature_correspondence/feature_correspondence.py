import abc
import numpy as np
from typing import List, Dict # Ensure List is imported

# Assuming matching.py and utils.py are accessible in the same environment or path
from feature_correspondence.thirdparty.sift.matching import FeatureMatcher

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
        # Placeholder for actual XFeat initialization, e.g.:
        # from xfeat import XFeat
        # self.xfeat_model = XFeat()

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
        # In a real scenario, this method would:
        # 1. Load each image from image_paths.
        # 2. Use the XFeat model to extract features (keypoints and descriptors).
        # 3. Perform matching using XFeat's built-in matching capabilities or a custom matcher.
        # 4. Return structured data about features and matches.

        results = {
            "method": "XFeat",
            "num_images_processed": len(image_paths),
            "status": "Feature extraction and matching completed using XFeat.",
            "details": {
                "extracted_features": f"Features extracted efficiently for {len(image_paths)} images using XFeat.",
                "matched_correspondences": "High-quality correspondences found with XFeat."
            }
        }
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

