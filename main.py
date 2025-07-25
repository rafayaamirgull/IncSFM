import os
import glob
import math
import numpy as np
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

# Local imports (ensure these files are in the same directory or properly installed)
import bundle_adjustment as b
# Removed direct import of FeatureMatcher, now handled by the factory
from reconstruction import ReconstructionPipeline

# Import BoVW functions and constants from utils
from utils import (
    get_images,
    export_to_colmap,
    visualize_sfm_and_pose_open3d,
    build_vocabulary,
    get_raw_bovw_histogram,
    calculate_idf,
    apply_tfidf_weighting,
    match_images_bovw_tfidf,
    VOCABULARY_SIZE,
)

# Import the FeatureExtractorMatcherFactory
from feature_correspondence.feature_correspondence import FeatureExtractorMatcherFactory


# --- Configuration and Constants ---
# Set up logging for better output control
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Default intrinsic camera matrix for templeRing
DEFAULT_K_MATRIX_STR = (
    "[[1520.40, 0.00, 302.32], [0.00, 1525.90, 246.87], [0.00, 0.00, 1.00]]"
)
MIN_PNP_POINTS_THRESHOLD = 6
BA_CHECKPOINTS_MULTIPLIER = 1.34
BA_CHECKPOINTS_START = [3, 4, 5, 6]
GLOBAL_BA_FTOL = 1e-3
GLOBAL_BA_N_ITER_TORCH = 1000
GLOBAL_BA_LEARNING_RATE_TORCH = 5e-5
REPROJECTION_THRESHOLD_PNP = 5
PNP_ITERATIONS = 200
LOCAL_BA_N_ITER_TORCH = 700
LOCAL_BA_LEARNING_RATE_TORCH = 1e-45
VOXEL_VISUALIZATION_THRESHOLD = (
    100  # Max sum of absolute coords for visualization filtering
)
DEFAULT_VOXELS = 100  # Set to 100 for faster visualization, 200 for higher resolution.


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the SfM pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Structure from Motion (SfM) pipeline."
    )

    parser.add_argument(
        "--use_pytorch_optimizer",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Set to False to use Scipy optimizer (slow) for Bundle Adjustment. Default: True",
    )
    parser.add_argument(
        "--show_plots_interactively",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Set to True to display plots interactively. Default: False",
    )
    parser.add_argument(
        "--save_plots",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Set to True to save generated plots to files. Default: False",
    )
    parser.add_argument(
        "--imgset",
        type=str,
        default="templeRing",
        help='Name of the image dataset (e.g., "templeRing", "dino"). Default: templeRing',
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=46,
        help="Number of images to process in the dataset. Default: 46",
    )
    parser.add_argument(
        "--k_matrix",
        type=str,
        default="[[1520.40, 0.00, 302.32], [0.00, 1525.90, 246.87], [0.00, 0.00, 1.00]]",
        help="Camera intrinsic matrix K as a string representing a 3x3 array. "
        'Example: "[[1520.40, 0.00, 302.32], [0.00, 1525.90, 246.87], [0.00, 0.00, 1.00]]"',
    )
    parser.add_argument(
        "--img_type",
        type=str,
        default="png",
        help='Image file extension (e.g., "png", "jpg", "JPG"). Default: png',
    )
    parser.add_argument(
        "--use_bovw_for_initial_pair",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Set to True to use BoVW for selecting the initial image pair. Default: True",
    )
    parser.add_argument(
        "--use_bovw_for_next_best_pair",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Set to True to use BoVW for selecting the next best image pair. Default: True",
    )
    parser.add_argument(
        "--feature_method",
        type=str,
        default="sift",
        help="Feature extraction and matching method to use ('sift' or 'xfeat'). Default: sift",
    )


    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")
    return args


def parse_k_matrix(k_matrix_str: str) -> np.ndarray:
    """Converts a string representation of the K matrix to a NumPy array."""
    try:
        k_list = eval(k_matrix_str)  # Safely evaluate the string as a Python literal
        K = np.array(k_list)
        if K.shape != (3, 3):
            raise ValueError("K matrix must be a 3x3 array.")
        return K
    except (SyntaxError, ValueError) as e:
        logging.error(
            f"Error parsing K matrix: {e}. Please ensure it's a valid 3x3 array string."
        )
        exit(1)  # Exit if K matrix is invalid


def visualize_reconstruction(
    points_3d_with_views: list,
    R_mats: dict,
    t_vecs: dict,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    base_path: str,
    imgset: str,
    img_type: str,
    all_keypoints: dict,
):
    """
    Prepares data and visualizes the 3D reconstruction and camera poses.
    """
    x, y, z = [], [], []
    for pt3 in points_3d_with_views:
        # Filter points for visualization to exclude outliers or very distant points
        if np.sum(np.abs(pt3.point3d[0])) < VOXEL_VISUALIZATION_THRESHOLD:
            x.append(pt3.point3d[0][0])
            y.append(pt3.point3d[0][1])
            z.append(pt3.point3d[0][2])

    vpoints = np.array(list(zip(x, y, z)))
    if vpoints.shape[0] == 0:
        logging.warning("No valid 3D points found for visualization after filtering.")
        return

    logging.info("Reconstruction finished. Exporting to COLMAP format...")

    colmap_output_dir = os.path.join(base_path, "colmap_export", imgset)
    os.makedirs(colmap_output_dir, exist_ok=True)  # Ensure output directory exists

    images_paths_for_export = sorted(
        glob.glob(
            os.path.join(base_path, "datasets", imgset, f"*.{img_type}"),
            recursive=True,
        )
    )
    # Ensure n_imgs is accurate or derived from len(images_paths_for_export)
    images_color_data = get_images(
        base_path, imgset, img_type, len(images_paths_for_export), type_="color"
    )

    chosen_color_strategy = "average"  # Or "first" or "median"

    export_to_colmap(
        output_path=colmap_output_dir,
        K_matrix=K,
        image_paths=images_paths_for_export,
        loaded_images=images_color_data,
        all_keypoints=all_keypoints,
        reconstructed_R_mats=R_mats,
        reconstructed_t_vecs=t_vecs,
        reconstructed_points3d_with_views=points_3d_with_views,
        image_height=img_h,
        image_width=img_w,
        point_color_strategy=chosen_color_strategy,
    )

    visualize_sfm_and_pose_open3d(
        points_3D=vpoints,
        camera_R_mats=R_mats,
        camera_t_vecs=t_vecs,
        K_matrix=K,
        image_width=img_w,
        image_height=img_h,
        frustum_scale=0.3,
        # point_colors=point_rgb_colors # point_rgb_colors is handled inside visualize_sfm_and_pose_open3d
    )
    logging.info("3D visualization complete.")


def main():
    """Main function to run the Structure from Motion (SfM) pipeline."""
    args = parse_arguments()

    USE_PYTORCH_OPTIMIZER = args.use_pytorch_optimizer
    SHOW_PLOTS_INTERACTIVELY = args.show_plots_interactively
    SAVE_PLOTS = args.save_plots
    imgset = args.imgset
    n_imgs = args.n_imgs
    img_type = args.img_type
    USE_BOVW_FOR_INITIAL_PAIR = args.use_bovw_for_initial_pair
    USE_BOVW_FOR_NEXT_BEST_PAIR = args.use_bovw_for_next_best_pair
    feature_method = args.feature_method # Get the chosen feature method
    print("Using following settings:    ", args)

    K = parse_k_matrix(args.k_matrix)

    base_path = os.getcwd()
    dataset_path = os.path.join(base_path, "datasets", imgset)
    if not os.path.exists(dataset_path):
        logging.error(
            f"Dataset path not found: {dataset_path}. Please check --imgset argument and directory structure."
        )
        exit(1)

    # Load images in grayscale for feature extraction
    images_gray = get_images(base_path, imgset, img_type, n_imgs, "gray")
    # Load images in color for plotting purposes (if save_plots is True)
    images_color = get_images(base_path, imgset, img_type, n_imgs, "color")


    if not images_gray:
        logging.error("No images loaded. Exiting.")
        exit(1)

    img_h, img_w = images_gray[0].shape[:2]
    logging.info(
        f"\n======== Using total {len(images_gray)} images of dataset {imgset} (Resolution: {img_w}x{img_h}) ========\n"
    )

    # --- Step 1: Feature Extraction and Matching using Factory Method ---
    logging.info(f"Initializing feature extractor/matcher for method: {feature_method}")
    try:
        # Use the factory to create the appropriate feature extractor and matcher object
        extractor_matcher = FeatureExtractorMatcherFactory.create_extractor_matcher(feature_method)

        # Call the process method on the created object
        # Pass the loaded grayscale images and plotting parameters
        processed_data = extractor_matcher.process(
            images_gray,
            save_plot=SAVE_PLOTS,
            dataset_name=imgset,
            images_color_for_plotting=images_color
        )
        logging.info(f"Feature extraction and matching completed using {processed_data['method']}.")

    except ValueError as e:
        logging.error(f"Error initializing feature matcher: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during feature processing: {e}")
        exit(1)


    # Extract all descriptors for BoVW processing
    all_image_descriptors = processed_data["descriptors"]

    if not all_image_descriptors:
        logging.error(
            "No descriptors found for any image. Cannot proceed with BoVW or SfM. Exiting."
        )
        exit(1)

    # --- Step 2: Build BoVW Vocabulary, Calculate IDF, and Generate TF-IDF Histograms ---
    vocabulary = None
    tfidf_image_histograms = {}

    if USE_BOVW_FOR_INITIAL_PAIR or USE_BOVW_FOR_NEXT_BEST_PAIR:
        logging.info("Building BoVW vocabulary and generating TF-IDF histograms...")
        vocab_size_ = int(VOCABULARY_SIZE * (1 + math.log10(n_imgs / 100)))
        logging.info(
            f"Total Images:    {n_imgs} \nBase vocabulary size: {VOCABULARY_SIZE} \nUpdated vocabulary size: {vocab_size_}"
        )
        vocabulary = build_vocabulary(all_image_descriptors, vocab_size=vocab_size_)

        if vocabulary is None:
            logging.error("Failed to build BoVW vocabulary. Disabling BoVW features.")
            USE_BOVW_FOR_INITIAL_PAIR = False  # Disable BoVW if vocabulary failed
            USE_BOVW_FOR_NEXT_BEST_PAIR = False

    if USE_BOVW_FOR_INITIAL_PAIR or USE_BOVW_FOR_NEXT_BEST_PAIR:
        logging.info("\n--- Generating TF-IDF weighted BoVW Histograms ---")
        raw_image_histograms = {}
        for img_idx, descriptors in enumerate(all_image_descriptors):
            hist = get_raw_bovw_histogram(descriptors, vocabulary)
            raw_image_histograms[img_idx] = hist

        list_of_raw_hists = list(raw_image_histograms.values())
        if list_of_raw_hists:
            idf_vector = calculate_idf(list_of_raw_hists)
            for img_idx, raw_hist in raw_image_histograms.items():
                tfidf_hist = apply_tfidf_weighting(raw_hist, idf_vector)
                tfidf_image_histograms[img_idx] = tfidf_hist
            logging.info("TF-IDF histograms generated.")
        else:
            logging.warning(
                "No raw histograms to calculate IDF. TF-IDF will not be used for pair selection."
            )
            USE_BOVW_FOR_INITIAL_PAIR = False
            USE_BOVW_FOR_NEXT_BEST_PAIR = False

    # Pass tfidf_image_histograms to ReconstructionPipeline
    rec_pipeline = ReconstructionPipeline(
        processed_data["adjacency_matrix"],
        processed_data["filtered_matches"],
        processed_data["keypoints"],
        processed_data["connected_pairs"],
        K,
        tfidf_image_histograms if USE_BOVW_FOR_NEXT_BEST_PAIR else None,
        USE_BOVW_FOR_NEXT_BEST_PAIR,
    )

    best_pair = None
    if USE_BOVW_FOR_INITIAL_PAIR:
        logging.info(
            "\n--- Selecting Initial Image Pair using BoVW TF-IDF Similarity ---"
        )
        max_similarity = -1.0  # Cosine similarity ranges from -1 to 1

        candidate_pairs = []
        for i in range(len(images_gray)):
            for j in range(i + 1, len(images_gray)):
                # Only consider pairs that have geometric matches from FeatureMatcher
                if (i, j) in processed_data["filtered_matches"] and len(
                    processed_data["filtered_matches"][(i, j)]
                ) > 0:
                    candidate_pairs.append((i, j))

        if not candidate_pairs:
            logging.warning(
                "No candidate pairs with filtered matches found for BoVW initial selection. Falling back to default."
            )
            USE_BOVW_FOR_INITIAL_PAIR = False
        else:
            best_similarity_found = -float("inf")
            temp_best_pair = None
            for i, j in candidate_pairs:
                hist_i = tfidf_image_histograms.get(i)
                hist_j = tfidf_image_histograms.get(j)

                # Ensure both histograms exist and are not empty
                if (
                    hist_i is not None
                    and hist_j is not None
                    and hist_i.size > 0
                    and hist_j.size > 0
                ):
                    similarity = match_images_bovw_tfidf(
                        hist_i, hist_j, metric="cosine"
                    )
                    if similarity > best_similarity_found:
                        best_similarity_found = similarity
                        temp_best_pair = (i, j)
            best_pair = temp_best_pair  # Assign the best found pair

        if best_pair is None:  # If BoVW still didn't find one, or no good matches
            logging.warning(
                "BoVW initial pair selection failed to find a suitable pair. Falling back to default."
            )
            USE_BOVW_FOR_INITIAL_PAIR = False

    if not USE_BOVW_FOR_INITIAL_PAIR:
        logging.info(
            "\n--- Selecting Initial Image Pair using default (feature match count) strategy ---"
        )
        best_pair_list = sorted(
            rec_pipeline.best_img_pair(top_x_perc=0.2)
        )  # Use original method
        if isinstance(best_pair_list, list) and len(best_pair_list) > 0:
            best_pair = tuple(best_pair_list)  # Ensure it's a tuple for consistency
        else:
            logging.error(
                "Failed to find initial image pair using default method. Exiting."
            )
            exit(1)

    logging.info(f"Selected initial image pair: {best_pair}")

    # Handle case where initialize_reconstruction returns None
    R0, t0, R1, t1, points3d_with_views = rec_pipeline.initialize_reconstruction(
        best_pair[0], best_pair[1]
    )

    if R0 is None or t0 is None or R1 is None or t1 is None or not points3d_with_views:
        logging.error(f"Initial reconstruction failed for pair {best_pair}. Exiting.")
        exit(1)

    R_mats = {best_pair[0]: R0, best_pair[1]: R1}
    t_vecs = {best_pair[0]: t0, best_pair[1]: t1}

    resected_imgs = [best_pair[0], best_pair[1]]
    unresected_imgs = [i for i in range(len(images_gray)) if i not in resected_imgs]
    logging.info(f"Initial image pair: {resected_imgs}")

    # Calculate Bundle Adjustment checkpoints dynamically
    ba_checkpoints = BA_CHECKPOINTS_START + [
        int(6 * (BA_CHECKPOINTS_MULTIPLIER**i)) for i in range(int(n_imgs / 2))
    ]
    # Remove duplicates and sort
    ba_checkpoints = sorted(list(set([cp for cp in ba_checkpoints if cp < n_imgs])))
    logging.info(
        f"Bundle Adjustment checkpoints set at: {ba_checkpoints} resected images."
    )

    iter_count = 0
    avg_errors_per_iteration = []

    while len(unresected_imgs) > 0:
        iter_count += 1
        logging.info(
            f"--- Iteration {iter_count}: {len(resected_imgs)} resected images, {len(unresected_imgs)} remaining ---"
        )

        resected_idx, unresected_idx, prepend = (
            rec_pipeline.next_img_pair_to_grow_reconstruction_scored(
                n_imgs,
                resected_imgs,
                unresected_imgs,
                rec_pipeline.img_adjacency,
                rec_pipeline.matches,
                rec_pipeline.keypoints,
                points3d_with_views,  # Pass points3d_with_views which now includes updated 3D points
            )
        )

        if unresected_idx is None or resected_idx is None:
            logging.info(
                "No more suitable image pairs could be found by the selection strategy. Ending reconstruction."
            )
            break  # Exit the while loop

        points3d_with_views, pts3d_for_pnp, pts2d_for_pnp, triangulation_status = (
            rec_pipeline.get_correspondences_for_pnp(
                resected_idx,
                unresected_idx,
                points3d_with_views,
                processed_data["filtered_matches"],
                processed_data["keypoints"],
            )
        )

        if (
            len(pts3d_for_pnp) < MIN_PNP_POINTS_THRESHOLD
            or len(pts2d_for_pnp) < MIN_PNP_POINTS_THRESHOLD
        ):
            logging.warning(
                f"Found only {len(pts3d_for_pnp)} 3D points and {len(pts2d_for_pnp)} 2D points. "
                f"Too few correspondences for PnP between unresected image {unresected_idx} "
                f"and resected image {resected_idx}. Skipping this attempt."
            )
            if unresected_idx in unresected_imgs:
                unresected_imgs.remove(unresected_idx)
                logging.info(
                    f"Image {unresected_idx} removed from unresected queue due to insufficient PnP points."
                )
            continue

        R_res = R_mats[resected_idx]
        t_res = t_vecs[resected_idx]
        logging.info(
            f"Attempting PnP for unresected image: {unresected_idx}, using resected: {resected_idx}"
        )

        R_new, t_new = rec_pipeline.do_pnp(
            pts3d_for_pnp,
            pts2d_for_pnp,
            K,
            iterations=PNP_ITERATIONS,
            reprojThresh=REPROJECTION_THRESHOLD_PNP,
        )

        if R_new is None or t_new is None:
            logging.warning(
                f"PnP failed for unresected image {unresected_idx} using resected image {resected_idx}. Skipping."
            )
            if unresected_idx in unresected_imgs:
                unresected_imgs.remove(unresected_idx)
                logging.info(
                    f"Image {unresected_idx} removed from unresected queue due to PnP failure."
                )
            continue

        R_mats[unresected_idx] = R_new
        t_vecs[unresected_idx] = t_new

        if prepend:
            resected_imgs.insert(0, unresected_idx)
        else:
            resected_imgs.append(unresected_idx)
        unresected_imgs.remove(unresected_idx)
        logging.info(
            f"Image {unresected_idx} successfully resected. Resected images: {resected_imgs}"
        )

        pnp_errors, projpts, avg_err, perc_inliers = (
            rec_pipeline.test_reproj_pnp_points(
                pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K
            )
        )
        logging.info(
            f"PnP reprojection error for image {unresected_idx}: {avg_err:.2f} pixels, Inliers: {perc_inliers*100:.2f}% ({len(pnp_errors)} pts)"
        )

        # Triangulation
        # Check if new triangulation is needed or if points need updating
        if (
            triangulation_status is not None and np.sum(triangulation_status) > 0
        ):  # Only triangulate if there are new matches

            # Get aligned keypoints for triangulation. The order of resected_idx, unresected_idx matters for kpts1, kpts2.
            # get_aligned_kpts expects (idx1, idx2) for matches[idx1][idx2]

            # The current rec_pipeline.get_correspondences_for_pnp populates `triangulation_status`
            # based on matches between resected_idx and unresected_idx.
            # So, kpts1 should correspond to resected_idx, kpts2 to unresected_idx.
            kpts_resected, kpts_unresected, kpts_resected_idxs, kpts_unresected_idxs = (
                rec_pipeline.get_aligned_kpts(
                    resected_idx,
                    unresected_idx,
                    processed_data["keypoints"],
                    processed_data["filtered_matches"],
                    mask=triangulation_status,
                )
            )

            if kpts_resected is not None and kpts_unresected is not None:
                points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = (
                    rec_pipeline.triangulate_points_and_reproject(
                        R_res,
                        t_res,
                        R_new,
                        t_new,
                        K,
                        points3d_with_views,
                        resected_idx,
                        unresected_idx,
                        kpts_resected,
                        kpts_unresected,
                        kpts_resected_idxs,
                        kpts_unresected_idxs,
                        compute_reproj=True,
                    )
                )
                logging.info(
                    f"Triangulation errors (L/R): {avg_tri_err_l:.2f}/{avg_tri_err_r:.2f} pixels."
                )
            else:
                logging.warning(
                    f"Skipping triangulation between {resected_idx} and {unresected_idx}: No valid aligned keypoints."
                )

        # Incremental Bundle Adjustment
        perform_ba = False
        # Condition for aggressive BA: high error or low inliers
        if (
            perc_inliers < 0.9 or avg_err > REPROJECTION_THRESHOLD_PNP * 1.5
        ):  # More aggressive threshold for BA trigger
            logging.info(
                "Intermediate BA triggered due to significant reprojection error or low inliers."
            )
            perform_ba = True
        elif len(resected_imgs) in ba_checkpoints:
            logging.info(
                f"Intermediate BA triggered at checkpoint {len(resected_imgs)} resected images."
            )
            perform_ba = True
        elif len(unresected_imgs) == 0:
            logging.info("Final BA triggered as all images are resected.")
            perform_ba = True

        if perform_ba:
            # Adjust ftol based on current error, lower ftol for higher accuracy when errors are low
            ftol_val = (
                1e-1 if perc_inliers <= 0.8 or avg_err >= 10 else 1e-2
            )  # Tighter ftol for better results
            logging.info(
                f"Performing intermediate Bundle Adjustment with ftol={ftol_val}..."
            )
            if not USE_PYTORCH_OPTIMIZER:
                points3d_with_views, R_mats, t_vecs = b.do_BA(
                    points3d_with_views,
                    R_mats,
                    t_vecs,
                    resected_imgs,
                    processed_data["keypoints"],
                    K,
                    ftol=ftol_val,
                )
            else:
                points3d_with_views, R_mats, t_vecs = b.do_BA_pytorch(
                    points3d_with_views,
                    R_mats,
                    t_vecs,
                    resected_imgs,
                    processed_data["keypoints"],
                    K,
                    n_iterations=LOCAL_BA_N_ITER_TORCH,
                    learning_rate=LOCAL_BA_LEARNING_RATE_TORCH,
                )

        # Recalculate average reprojection errors for all resected images after BA
        current_avg_reproj_errors = []
        for im_idx in resected_imgs:
            _, _, avg_error, _ = rec_pipeline.get_reproj_errors(
                im_idx,
                points3d_with_views,
                R_mats[im_idx],
                t_vecs[im_idx],
                K,
                processed_data["keypoints"],
                distCoeffs=np.array([]),
            )
            logging.info(
                f"Average reprojection error on image {im_idx} is {avg_error:.2f} pixels"
            )
            current_avg_reproj_errors.append(avg_error)
        overall_avg_error = (
            np.mean(current_avg_reproj_errors) if current_avg_reproj_errors else 0
        )
        logging.info(
            f"Average reprojection error across all {len(resected_imgs)} resected images is {overall_avg_error:.2f} pixels"
        )
        avg_errors_per_iteration.append(overall_avg_error)

    logging.info("\n--- Incremental Reconstruction Complete ---")

    # Plotting average reprojection errors
    if SHOW_PLOTS_INTERACTIVELY or SAVE_PLOTS:
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(avg_errors_per_iteration) + 1),
            avg_errors_per_iteration,
            marker="o",
        )
        plt.title("Average Reprojection Error per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reprojection Error (pixels)")
        plt.grid(True)
        if SHOW_PLOTS_INTERACTIVELY:
            plt.show()
        if SAVE_PLOTS:
            plot_dir = os.path.join(base_path, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{imgset}_avg_reprojection_errors.png"))
            logging.info(
                f"Saved plot to {os.path.join(plot_dir, f'{imgset}_avg_reprojection_errors.png')}"
            )
        plt.close()

    if len(resected_imgs) > 2 and points3d_with_views:
        logging.info(
            f"\n======== Performing Final Global Bundle Adjustment on {len(resected_imgs)} cameras and {len(points3d_with_views)} points ========"
        )
        if not USE_PYTORCH_OPTIMIZER:
            points3d_with_views, R_mats, t_vecs = b.do_BA(
                points3d_with_views,
                R_mats,
                t_vecs,
                resected_imgs,
                processed_data["keypoints"],
                K,
                ftol=GLOBAL_BA_FTOL,
            )
        else:
            points3d_with_views, R_mats, t_vecs = b.do_BA_pytorch(
                points3d_with_views,
                R_mats,
                t_vecs,
                resected_imgs,
                processed_data["keypoints"],
                K,
                n_iterations=GLOBAL_BA_N_ITER_TORCH,
                learning_rate=GLOBAL_BA_LEARNING_RATE_TORCH,
            )
        logging.info("======== Global Bundle Adjustment Complete ========")
    else:
        logging.info(
            "Skipping Final Global Bundle Adjustment: Not enough resected images or 3D points."
        )

    # --- Final Visualization ---
    visualize_reconstruction(
        points3d_with_views,
        R_mats,
        t_vecs,
        K,
        img_w,
        img_h,
        base_path,
        imgset,
        img_type,
        processed_data["keypoints"],
    )


if __name__ == "__main__":
    main()
