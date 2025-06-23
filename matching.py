import cv2
import numpy as np
from typing import List, Tuple, Dict
import os
from utils import plot_and_save_keypoints, plot_and_save_matches


class FeatureMatcher:
    """
    A robust utility for extracting and matching SIFT features between images,
    including outlier rejection and connectivity analysis.
    """

    def __init__(
        self,
        ratio_threshold: float = 0.75,
        min_inlier_matches: int = 20,
        ransac_reprojection_threshold: float = 3.0,
    ):
        """
        Initializes the feature matcher with specified parameters.

        Args:
            ratio_threshold (float): The threshold for Lowe's ratio test,
                                     used to filter ambiguous matches.
            min_inlier_matches (int): The minimum number of geometrically
                                      consistent matches required to consider
                                      two images connected.
            ransac_reprojection_threshold (float): The maximum allowed
                                                   reprojection error for a point
                                                   to be considered an inlier during RANSAC
                                                   fundamental matrix estimation.
        """
        # Initialize SIFT feature detector and descriptor extractor
        self.sift = cv2.SIFT_create()
        # BFMatcher with L2 norm is suitable for SIFT descriptors
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio_threshold = ratio_threshold
        self.min_inlier_matches = min_inlier_matches
        self.ransac_reprojection_threshold = ransac_reprojection_threshold

    def extract_sift_features(
        self, images: List[np.ndarray]
    ) -> List[Tuple[List[cv2.KeyPoint], np.ndarray]]:
        """
        Detects SIFT keypoints and computes their descriptors for a list of images.

        Args:
            images (List[np.ndarray]): A list of grayscale input images.

        Returns:
            List[Tuple[List[cv2.KeyPoint], np.ndarray]]: A list where each element
                                                        is a tuple containing
                                                        (keypoints, descriptors)
                                                        for an image.
        """
        print("\n--- Extracting SIFT Features ---")
        image_features: List[Tuple[List[cv2.KeyPoint], np.ndarray]] = []
        for i, img in enumerate(images):
            keypoints, descriptors = self.sift.detectAndCompute(img, None)
            if descriptors is None:  # Handle cases where no features are found
                descriptors = np.array([])
            print(f"  Extracted {len(keypoints)} SIFT features from image #{i}")
            image_features.append((keypoints, descriptors))
        return image_features

    def find_raw_matches(
        self, image_descriptors: List[np.ndarray]
    ) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
        """
        Performs k-NN matching (k=2) between descriptors of all unique image pairs
        and applies Lowe's ratio test.

        Args:
            image_descriptors (List[np.ndarray]): A list of SIFT descriptor arrays,
                                                  one for each image.

        Returns:
            Dict[Tuple[int, int], List[cv2.DMatch]]: A dictionary where keys are
                                                      (image_idx1, image_idx2) tuples
                                                      and values are lists of
                                                      `cv2.DMatch` objects representing
                                                      the raw, ratio-tested matches.
        """
        print("\n--- Performing Feature Matching ---")
        num_images = len(image_descriptors)
        raw_matches_map: Dict[Tuple[int, int], List[cv2.DMatch]] = {}

        for i in range(num_images):
            for j in range(i + 1, num_images):
                desc1 = image_descriptors[i]
                desc2 = image_descriptors[j]
                # Skip if either image has no descriptors
                if desc1.size == 0 or desc2.size == 0:
                    print(
                        f"  Skipping match between image #{i} and #{j}: one or both have no descriptors."
                    )
                    continue

                # Perform k-NN matching
                knn_matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)

                raw_matches_map[(i, j)] = good_matches
                print(
                    f"  Image #{i} and Image #{j}: {len(good_matches)} raw matches (after ratio test)"
                )
        return raw_matches_map

    def filter_matches_geometric_consistency(
        self,
        raw_matches_map: Dict[Tuple[int, int], List[cv2.DMatch]],
        all_image_keypoints: List[List[cv2.KeyPoint]],
    ) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
        """
        Filters raw matches using RANSAC to estimate the Fundamental Matrix,
        retaining only geometrically consistent inliers.

        Args:
            raw_matches_map (Dict[Tuple[int, int], List[cv2.DMatch]]): Dictionary
                                                                      of raw matches
                                                                      between image pairs.
            all_image_keypoints (List[List[cv2.KeyPoint]]): A list of keypoints
                                                            for each image.

        Returns:
            Dict[Tuple[int, int], List[cv2.DMatch]]: A dictionary containing only
                                                      the inlier matches that pass
                                                      the RANSAC filtering.
        """
        print("\n--- Filtering Matches for Geometric Consistency (RANSAC) ---")
        filtered_matches_map: Dict[Tuple[int, int], List[cv2.DMatch]] = {}

        for (idx1, idx2), matches_list in raw_matches_map.items():
            filtered_matches_map[(idx1, idx2)] = []
            if len(matches_list) < self.min_inlier_matches:
                print(
                    f"  Skipping RANSAC for Image #{idx1} and #{idx2}: too few raw matches ({len(matches_list)})"
                )
                continue

            # Extract keypoint coordinates for matched points
            src_pts = np.float32(
                [all_image_keypoints[idx1][m.queryIdx].pt for m in matches_list]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [all_image_keypoints[idx2][m.trainIdx].pt for m in matches_list]
            ).reshape(-1, 1, 2)

            # Estimate Fundamental Matrix using RANSAC
            fundamental_matrix, mask = cv2.findFundamentalMat(
                src_pts,
                dst_pts,
                cv2.FM_RANSAC,
                self.ransac_reprojection_threshold,
                0.99,
            )

            # Filter out matches based on the RANSAC mask
            if (
                mask is None
                or fundamental_matrix is None
                or (
                    fundamental_matrix.shape == (3, 3)
                    and np.linalg.det(fundamental_matrix) > 1e-7
                )
            ):
                print(
                    f"  Image #{idx1} and #{idx2}: RANSAC failed to find a reliable Fundamental Matrix."
                )
                continue

            inlier_matches = [
                matches_list[i] for i in range(len(matches_list)) if mask[i] == 1
            ]

            if len(inlier_matches) >= self.min_inlier_matches:
                filtered_matches_map[(idx1, idx2)] = inlier_matches
                print(
                    f"  Image #{idx1} and #{idx2}: {len(inlier_matches)} inlier matches (after RANSAC)"
                )
            else:
                print(
                    f"  Image #{idx1} and #{idx2}: Not enough inlier matches ({len(inlier_matches)}) after RANSAC."
                )

        return filtered_matches_map

    def build_connectivity_graph(
        self, num_images: int, filtered_matches: Dict[Tuple[int, int], List[cv2.DMatch]]
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Constructs an adjacency matrix and a list of connected pairs based on
        the filtered inlier matches.

        Args:
            num_images (int): The total number of images in the dataset.
            filtered_matches (Dict[Tuple[int, int], List[cv2.DMatch]]): A dictionary
                                                                       of geometrically
                                                                       consistent matches.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: A tuple containing:
                - np.ndarray: The adjacency matrix, where `adjacency[i, j] = 1`
                              if images `i` and `j` are connected, else `0`.
                - List[Tuple[int, int]]: A list of tuples `(i, j)` representing
                                         all connected image pairs.
        """
        print("\n--- Building Image Connectivity Graph ---")
        adjacency_matrix = np.zeros((num_images, num_images), dtype=int)
        connected_image_pairs: List[Tuple[int, int]] = []

        for (idx1, idx2), matches in filtered_matches.items():
            if matches:  # If there are any inlier matches
                adjacency_matrix[idx1, idx2] = 1
                adjacency_matrix[idx2, idx1] = 1  # Graph is undirected
                connected_image_pairs.append((idx1, idx2))
                print(f"  Connection established between image #{idx1} and #{idx2}")

        return adjacency_matrix, connected_image_pairs

    def saving_plots_kpm(
        self,
        num_images,
        dataset_name,
        all_keypoints,
        filtered_matches,
        images,
        images_color_for_plotting,
    ):
        # --- Create Output Directories ---
        output_plots_base_dir = os.path.join(os.getcwd(), "output_plots", dataset_name)
        features_out_dir = os.path.join(output_plots_base_dir, "features")
        matches_out_dir = os.path.join(output_plots_base_dir, "feature_matches")
        os.makedirs(features_out_dir, exist_ok=True)
        os.makedirs(matches_out_dir, exist_ok=True)
        print(f"Saving feature plots to: {features_out_dir}")
        print(f"Saving match plots to: {matches_out_dir}")
        print(
            f"\n\n\n======== Plotting and Saving Features (first few images as example) ========"
        )

        for i in range(num_images):
            img_for_plot = (
                images_color_for_plotting[i] if images_color_for_plotting else images[i]
            )
            plot_and_save_keypoints(
                img_for_plot,
                all_keypoints[i],
                os.path.join(features_out_dir, f"features_img_{i:03d}.png"),
                title=f"Detected {len(all_keypoints[i])} SIFT Features in Image {i}",
                show_plot=False,
            )

        print(
            f"\n======== Plotting sssand Saving Filtered Matches (example pairs) ========"
        )
        num_matches_to_plot = 0
        for i in range(num_images):
            for j in range(i + 1, num_images):
                if filtered_matches[i, j] and len(filtered_matches[i, j]) > 0:
                    img1_for_plot = (
                        images_color_for_plotting[i]
                        if images_color_for_plotting
                        else images[i]
                    )
                    img2_for_plot = (
                        images_color_for_plotting[j]
                        if images_color_for_plotting
                        else images[j]
                    )

                    plot_and_save_matches(
                        img1_for_plot,
                        all_keypoints[i],
                        img2_for_plot,
                        all_keypoints[j],
                        filtered_matches[i, j],  # Draw the good (filtered) matches
                        os.path.join(
                            matches_out_dir, f"matches_{i:03d}_vs_{j:03d}.png"
                        ),
                        title=f"Filtered Matches {len(filtered_matches[i,j])} between Image {i} and Image {j}",
                        show_plot=False,
                    )
                    num_matches_to_plot += 1

    def process_images(
        self,
        images: List[np.ndarray],
        save_plot=False,
        dataset_name="",
        images_color_for_plotting: List[np.ndarray] = [],
    ) -> Dict[str, any]:
        """
        Orchestrates the entire feature matching pipeline:
        feature extraction, raw matching, outlier filtering, and connectivity analysis.

        Args:
            images (List[np.ndarray]): A list of grayscale images to process.

        Returns:
            Dict[str, any]: A dictionary containing the results:
                - 'keypoints_descriptors': List of (keypoints, descriptors) for each image.
                - 'raw_matches': Dictionary of raw, ratio-tested matches.
                - 'filtered_matches': Dictionary of geometrically consistent matches.
                - 'adjacency_matrix': NxN numpy array indicating image connectivity.
                - 'connected_pairs': List of (i, j) tuples for connected image pairs.
                - 'total_inlier_matches': Total count of all filtered matches.
        """
        if not images:
            print("No images provided for processing.")
            return {}

        # Step 1: Extract SIFT features
        keypoints_descriptors = self.extract_sift_features(images)
        all_keypoints = [kp for kp, _ in keypoints_descriptors]
        all_descriptors = [des for _, des in keypoints_descriptors]

        # Step 2: Find raw matches between all image pairs
        raw_matches = self.find_raw_matches(all_descriptors)

        # Step 3: Filter matches using RANSAC for geometric consistency
        filtered_matches = self.filter_matches_geometric_consistency(
            raw_matches, all_keypoints
        )

        # Step 4: Build connectivity graph
        num_images = len(images)
        adjacency_matrix, connected_pairs = self.build_connectivity_graph(
            num_images, filtered_matches
        )

        # Count total inlier matches
        total_inlier_matches = sum(len(m) for m in filtered_matches.values())
        print(
            f"\n--- Total Inlier Matches Across All Pairs: {total_inlier_matches} ---"
        )

        if save_plot:
            self.saving_plots_kpm(
                num_images,
                dataset_name,
                all_keypoints,
                filtered_matches,
                images,
                images_color_for_plotting,
            )

        return {
            "keypoints": all_keypoints,
            "descriptors": all_descriptors,
            "filtered_matches": filtered_matches,
            "adjacency_matrix": adjacency_matrix,
            "connected_pairs": connected_pairs,
            "total_inlier_matches": total_inlier_matches,
        }
