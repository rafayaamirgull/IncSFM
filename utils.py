import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from typing import Tuple, List, Dict, Optional
import os
import glob
from sklearn.cluster import MiniBatchKMeans  # Efficient for large datasets
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

# --- BoVW Configuration and Functions ---
# NOTE: Ensure FEATURE_DETECTOR_TYPE here matches the detector used in your FeatureMatcher class
# (e.g., if FeatureMatcher uses SIFT, set to "SIFT"; if ORB, set to "ORB").
FEATURE_DETECTOR_TYPE = "SIFT"  # Set to match your FeatureMatcher's detector
VOCABULARY_SIZE = 1000  # Number of visual words (clusters)
BATCH_SIZE = 256  # For MiniBatchKMeans, processes data in mini-batches
RANDOM_STATE = 42  # For reproducibility of clustering


def build_vocabulary(
    all_descriptors_list: List[np.ndarray], vocab_size: int = VOCABULARY_SIZE
) -> MiniBatchKMeans:
    """
    Builds a visual vocabulary (codebook) by clustering descriptors from a list of descriptor arrays.

    Args:
        all_descriptors_list (list): A list of NumPy arrays, where each array contains descriptors
                                     from one image.
        vocab_size (int): The desired number of visual words.

    Returns:
        MiniBatchKMeans: The trained KMeans model representing the visual vocabulary.
    """
    valid_descriptors = [
        d for d in all_descriptors_list if d is not None and d.shape[0] > 0
    ]
    if not valid_descriptors:
        print(
            "Error: No valid descriptors found to build vocabulary. Please check feature extraction."
        )
        return None

    # Concatenate all descriptors into a single NumPy array
    all_descriptors_flat = np.vstack(valid_descriptors)

    print(
        f"Clustering {all_descriptors_flat.shape[0]} descriptors into {vocab_size} visual words..."
    )
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        n_init="auto",
        verbose=False,
    )
    kmeans.fit(all_descriptors_flat)
    print("Vocabulary built successfully.")
    return kmeans


def get_raw_bovw_histogram(
    descriptors: np.ndarray, vocabulary_model: MiniBatchKMeans
) -> np.ndarray:
    """
    Represents an image as a raw Bag of Visual Words histogram (Term Frequency).
    This histogram is not normalized yet.

    Args:
        descriptors (np.ndarray): Descriptors for a single image.
        vocabulary_model (MiniBatchKMeans): The trained KMeans model (visual vocabulary).

    Returns:
        np.ndarray: A raw histogram of visual word occurrences for the image.
    """
    if descriptors is None or descriptors.shape[0] == 0:
        return np.zeros(
            vocabulary_model.n_clusters
        )  # Return empty histogram if no features

    # Predict which cluster (visual word) each descriptor belongs to
    visual_words = vocabulary_model.predict(descriptors)

    # Create histogram (term frequency)
    histogram = np.zeros(vocabulary_model.n_clusters)
    for word_index in visual_words:
        histogram[word_index] += 1
    return histogram


def calculate_idf(all_raw_histograms: List[np.ndarray]) -> np.ndarray:
    """
    Calculates the Inverse Document Frequency (IDF) for each visual word.

    Args:
        all_raw_histograms (list of np.ndarray): A list of raw BoVW histograms for all images.

    Returns:
        np.ndarray: An array of IDF values, one for each visual word.
    """
    num_documents = len(all_raw_histograms)
    if num_documents == 0:
        return np.array([])  # Return empty if no documents

    # Initialize document_frequency based on the vocabulary size
    vocab_size = all_raw_histograms[0].shape[0]
    document_frequency = np.zeros(vocab_size)

    for hist in all_raw_histograms:
        # Increment document_frequency for words present in the current histogram
        document_frequency[hist > 0] += 1

    # Add 1 to denominator to avoid division by zero for words not appearing in any document
    idf = np.log(num_documents / (1 + document_frequency))
    return idf


def apply_tfidf_weighting(
    raw_histogram: np.ndarray, idf_vector: np.ndarray
) -> np.ndarray:
    """
    Applies TF-IDF weighting to a raw BoVW histogram.

    Args:
        raw_histogram (np.ndarray): The raw (term frequency) BoVW histogram of an image.
        idf_vector (np.ndarray): The pre-calculated IDF vector.

    Returns:
        np.ndarray: The TF-IDF weighted and L2-normalized histogram.
    """
    # Check for empty idf_vector (e.g., if no images or vocabulary failed)
    if idf_vector.size == 0 or raw_histogram.size == 0:
        return np.zeros_like(raw_histogram)

    # Apply TF-IDF weighting
    tfidf_histogram = raw_histogram * idf_vector

    # L2 normalize the TF-IDF histogram. This is standard practice for TF-IDF vectors
    # for similarity comparison.
    norm = np.linalg.norm(tfidf_histogram)
    if norm > 0:
        tfidf_histogram = tfidf_histogram / norm
    return tfidf_histogram


def match_images_bovw_tfidf(
    hist1: np.ndarray, hist2: np.ndarray, metric: str = "cosine"
) -> float:
    """
    Compares two TF-IDF weighted BoVW histograms to determine their similarity.

    Args:
        hist1 (np.ndarray): The TF-IDF BoVW histogram of the first image.
        hist2 (np.ndarray): The TF-IDF BoVW histogram of the second image.
        metric (str): The similarity metric to use ("cosine" or "euclidean").

    Returns:
        float: A similarity score. Higher means more similar for cosine, lower for euclidean.
    """
    # Handle empty histograms, assume no similarity
    if hist1.size == 0 or hist2.size == 0:
        return (
            0.0 if metric == "cosine" else float("inf")
        )  # 0 for cosine (no similarity), inf for euclidean (max distance)

    if metric == "cosine":
        # Cosine similarity ranges from -1 to 1. Higher is more similar.
        return cosine_similarity(hist1.reshape(1, -1), hist2.reshape(1, -1))[0][0]
    elif metric == "euclidean":
        # Euclidean distance. Lower is more similar.
        return euclidean_distances(hist1.reshape(1, -1), hist2.reshape(1, -1))[0][0]
    else:
        raise ValueError(
            f"Unsupported metric: {metric}. Choose 'cosine' or 'euclidean'."
        )


def get_images(base_path, dataset_path, img_format, use_n_imgs=-1, type_="color"):
    images_paths = sorted(
        glob.glob(
            os.path.join(base_path, "datasets", dataset_path) + "/*." + img_format,
            recursive=True,
        )
    )
    images = []
    if not use_n_imgs == -1 and use_n_imgs <= len(images_paths):
        images_paths = images_paths[:use_n_imgs]

    for img_path in images_paths:
        if type_ == "color":
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        elif type_ == "gray":
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid image type. Choose 'color' or 'gray'.")
        images.append(image)
    return images


def build_histogram(descriptor, kmeans):
    labels = kmeans.predict(descriptor)
    histogram, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
    return histogram / np.sum(histogram)  # Normalize histogram


class Point3DWithViews:
    """
    Represents a 3D point along with the indices of its corresponding 2D keypoints in multiple images.

    Attributes:
        point3d (np.ndarray): The 3D coordinates of the point (shape: (3,)).
        source_2dpt_idxs (Dict[int, int]): Mapping from image index to keypoint index within that image.
    """

    def __init__(self, point3d: np.ndarray, source_2dpt_idxs: Dict[int, int]):
        self.point3d = point3d
        self.source_2dpt_idxs = source_2dpt_idxs


# Helper function to convert rotation matrix to quaternion (w, x, y, z)
# This is a common implementation. You might need to adjust based on COLMAP's exact convention
# if it differs, but (w, x, y, z) for R_cw (camera from world) is typical.
def rotation_matrix_to_quaternion(R):
    """Converts a rotation matrix to a quaternion (w, x, y, z)."""
    # Ensure the matrix is a NumPy array
    R = np.asarray(R, dtype=float)

    # Calculate the trace of the matrix
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return qw, qx, qy, qz


def _sample_bgr_bilinear(image: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
    """
    Bilinear sample in BGR image coordinates. Returns uint8 BGR vector or None if out of bounds.
    """
    if image is None or image.ndim != 3:
        return None
    h, w = image.shape[:2]
    if x < 0.0 or y < 0.0 or x > (w - 1) or y > (h - 1):
        return None

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    wx = x - x0
    wy = y - y0

    c00 = image[y0, x0].astype(np.float32)
    c10 = image[y0, x1].astype(np.float32)
    c01 = image[y1, x0].astype(np.float32)
    c11 = image[y1, x1].astype(np.float32)

    sampled = (
        (1.0 - wx) * (1.0 - wy) * c00
        + wx * (1.0 - wy) * c10
        + (1.0 - wx) * wy * c01
        + wx * wy * c11
    )
    return np.clip(np.round(sampled), 0, 255).astype(np.uint8)


def _compute_point_color_bgr(
    pt_obj: Point3DWithViews,
    loaded_images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    reconstructed_R_mats: Dict[int, np.ndarray],
    image_height: int,
    image_width: int,
    point_color_strategy: str,
) -> Optional[np.ndarray]:
    """
    Compute one BGR color for a 3D point from all its valid 2D observations.
    """
    point_colors_bgr_list = []
    sorted_observing_py_img_indices = sorted(pt_obj.source_2dpt_idxs.keys())

    for img_py_idx in sorted_observing_py_img_indices:
        if img_py_idx not in reconstructed_R_mats:
            continue
        kpt_original_idx = pt_obj.source_2dpt_idxs[img_py_idx]

        if not (0 <= img_py_idx < len(loaded_images)):
            continue
        if not (0 <= kpt_original_idx < len(all_keypoints[img_py_idx])):
            continue

        kp = all_keypoints[img_py_idx][kpt_original_idx]
        kp_x, kp_y = kp.pt
        if not (0.0 <= kp_y < image_height and 0.0 <= kp_x < image_width):
            continue

        bgr_pixel = _sample_bgr_bilinear(loaded_images[img_py_idx], kp_x, kp_y)
        if bgr_pixel is not None:
            point_colors_bgr_list.append(bgr_pixel)

    if not point_colors_bgr_list:
        return None

    stacked = np.stack(point_colors_bgr_list, axis=0).astype(np.float32)
    if point_color_strategy == "first":
        return stacked[0].astype(np.uint8)
    if point_color_strategy == "median":
        return np.clip(np.round(np.median(stacked, axis=0)), 0, 255).astype(np.uint8)

    # Default to average for robustness/noise smoothing.
    return np.clip(np.round(np.mean(stacked, axis=0)), 0, 255).astype(np.uint8)


def compute_point_colors_for_visualization(
    reconstructed_points3d_with_views: List[Point3DWithViews],
    loaded_images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    reconstructed_R_mats: Dict[int, np.ndarray],
    image_height: int,
    image_width: int,
    point_color_strategy: str = "average",
) -> np.ndarray:
    """
    Computes per-point RGB colors (0-1) for all reconstructed points in list order.
    """
    valid_color_strategies = ["first", "average", "median"]
    if point_color_strategy not in valid_color_strategies:
        raise ValueError(
            f"Invalid point_color_strategy. Choose from {valid_color_strategies}"
        )

    colors_rgb = []
    default_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    for pt_obj in reconstructed_points3d_with_views:
        bgr = _compute_point_color_bgr(
            pt_obj=pt_obj,
            loaded_images=loaded_images,
            all_keypoints=all_keypoints,
            reconstructed_R_mats=reconstructed_R_mats,
            image_height=image_height,
            image_width=image_width,
            point_color_strategy=point_color_strategy,
        )
        if bgr is None:
            colors_rgb.append(default_rgb)
        else:
            # Convert BGR [0,255] -> RGB [0,1]
            colors_rgb.append(np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32) / 255.0)

    return np.asarray(colors_rgb, dtype=np.float32)


def export_to_colmap(
    output_path: str,
    K_matrix: np.matrix,
    image_paths: List[str],
    loaded_images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    reconstructed_R_mats: Dict[int, np.ndarray],
    reconstructed_t_vecs: Dict[int, np.ndarray],
    reconstructed_points3d_with_views: List[Point3DWithViews],
    image_height: int,
    image_width: int,
    point_color_strategy: str = "first",  # NEW: "first", "average", or "median"
):
    """
    Exports reconstruction data to COLMAP text format, with selectable RGB strategy.
    """
    os.makedirs(output_path, exist_ok=True)
    valid_color_strategies = ["first", "average", "median"]
    if point_color_strategy not in valid_color_strategies:
        raise ValueError(
            f"Invalid point_color_strategy. Choose from {valid_color_strategies}"
        )

    # --- 1. cameras.txt (remains the same) ---
    with open(os.path.join(output_path, "cameras.txt"), "w") as f_cam:
        f_cam.write("# Camera list with one line of data per camera:\n")
        f_cam.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f_cam.write(f"# Number of cameras: 1\n")

        cam_id = 1
        model = "PINHOLE"
        width = image_width
        height = image_height
        fx = K_matrix[0, 0]
        fy = K_matrix[1, 1]
        cx = K_matrix[0, 2]
        cy = K_matrix[1, 2]
        f_cam.write(f"{cam_id} {model} {width} {height} {fx} {fy} {cx} {cy}\n")

    colmap_img_id_map = {
        py_idx: col_idx
        for col_idx, py_idx in enumerate(sorted(reconstructed_R_mats.keys()), 1)
    }
    # --- 2. points3D.txt (modified for RGB strategy) ---
    points3d_lines_buffer = []
    point_rgb_colors = []

    num_valid_3d_points = 0
    track_lengths = []

    for pt3d_py_idx, pt_obj in enumerate(reconstructed_points3d_with_views):
        colmap_point3d_id = pt3d_py_idx + 1

        x_3d, y_3d, z_3d = pt_obj.point3d.ravel()

        r_val, g_val, b_val = 0, 0, 0  # Default color
        bgr = _compute_point_color_bgr(
            pt_obj=pt_obj,
            loaded_images=loaded_images,
            all_keypoints=all_keypoints,
            reconstructed_R_mats=reconstructed_R_mats,
            image_height=image_height,
            image_width=image_width,
            point_color_strategy=point_color_strategy,
        )
        if bgr is not None:
            b_val, g_val, r_val = int(bgr[0]), int(bgr[1]), int(bgr[2])

        # Note: COLMAP expects R G B order
        final_r, final_g, final_b = r_val, g_val, b_val

        error = 0.0

        track_str_parts = []
        current_track_length = 0
        for img_py_idx, kpt_original_idx in pt_obj.source_2dpt_idxs.items():
            if img_py_idx in colmap_img_id_map:
                colmap_image_id = colmap_img_id_map[img_py_idx]
                track_str_parts.extend([str(colmap_image_id), str(kpt_original_idx)])
                current_track_length += 1

        if current_track_length >= 2:
            num_valid_3d_points += 1
            track_lengths.append(current_track_length)
            track_str = " ".join(track_str_parts)
            points3d_lines_buffer.append(
                f"{colmap_point3d_id} {x_3d} {y_3d} {z_3d} {final_r} {final_g} {final_b} {error} {track_str}\n"
            )
            point_rgb_colors.append([final_r / 255, final_g / 255, final_b / 255])
    with open(os.path.join(output_path, "points3D.txt"), "w") as f_pts3d:
        f_pts3d.write("# 3D point list with one line of data per point:\n")
        f_pts3d.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        mean_track_len_val = np.mean(track_lengths) if track_lengths else 0
        f_pts3d.write(
            f"# Number of points: {num_valid_3d_points}, mean track length: {mean_track_len_val}\n"
        )
        for line in points3d_lines_buffer:
            f_pts3d.write(line)

    # --- 3. images.txt ---
    images_lines_buffer = []
    total_observations_for_header = 0

    for py_img_idx in sorted(reconstructed_R_mats.keys()):
        if py_img_idx not in colmap_img_id_map:
            continue

        colmap_image_id = colmap_img_id_map[py_img_idx]
        R = reconstructed_R_mats[py_img_idx]
        t = reconstructed_t_vecs[py_img_idx].ravel()
        qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
        tx, ty, tz = t[0], t[1], t[2]
        camera_colmap_id = 1
        img_name = os.path.basename(image_paths[py_img_idx])

        images_lines_buffer.append(
            f"{colmap_image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_colmap_id} {img_name}\n"
        )

        points2d_line_parts = []
        num_observations_in_image = 0

        img_kpt_to_pt3d_id = {}
        for pt3d_py_idx, pt_obj in enumerate(reconstructed_points3d_with_views):
            temp_track_len = 0
            for obs_img_idx in pt_obj.source_2dpt_idxs.keys():
                if obs_img_idx in reconstructed_R_mats:
                    temp_track_len += 1

            if temp_track_len >= 2:
                colmap_pt3d_id_current = pt3d_py_idx + 1
                if py_img_idx in pt_obj.source_2dpt_idxs:
                    kpt_orig_idx = pt_obj.source_2dpt_idxs[py_img_idx]
                    img_kpt_to_pt3d_id[(py_img_idx, kpt_orig_idx)] = (
                        colmap_pt3d_id_current
                    )

        for kpt_original_idx, kp in enumerate(all_keypoints[py_img_idx]):
            x_2d, y_2d = kp.pt
            observed_colmap_point3d_id = img_kpt_to_pt3d_id.get(
                (py_img_idx, kpt_original_idx), -1
            )
            points2d_line_parts.extend(
                [str(x_2d), str(y_2d), str(observed_colmap_point3d_id)]
            )
            if observed_colmap_point3d_id != -1:
                num_observations_in_image += 1

        total_observations_for_header += num_observations_in_image
        images_lines_buffer.append(" ".join(points2d_line_parts) + "\n")

    with open(os.path.join(output_path, "images.txt"), "w") as f_img:
        f_img.write("# Image list with two lines of data per image:\n")
        f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_img.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        mean_obs_val = (
            total_observations_for_header / len(colmap_img_id_map)
            if colmap_img_id_map
            else 0
        )
        f_img.write(
            f"# Number of images: {len(colmap_img_id_map)}, mean observations per image: {mean_obs_val}\n"
        )
        for line in images_lines_buffer:
            f_img.write(line)

    print(
        f"COLMAP data exported to {output_path} using '{point_color_strategy}' color strategy."
    )
    return np.array(point_rgb_colors)


def plot_and_save_keypoints(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    output_path: str,
    title: str = "Detected Keypoints",
    show_plot: bool = False,
):
    """
    Draws keypoints on an image, saves the plot, and optionally displays it.

    Args:
        image (np.ndarray): The input image (grayscale or color).
        keypoints (List[cv2.KeyPoint]): A list of detected keypoints.
        output_path (str): The full path including filename where the plot will be saved.
                           E.g., "plots/image1_keypoints.png"
        title (str): The title for the plot.
        show_plot (bool): If True, displays the plot in a new window.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert grayscale image to BGR if it's 2D for cv2.drawKeypoints
    display_image = image
    if len(image.shape) == 2:  # Grayscale image
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw keypoints on a copy of the image
    image_with_kp = cv2.drawKeypoints(
        display_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(10, 8))
    # Convert BGR (OpenCV) to RGB (Matplotlib) if the image is color
    if len(image_with_kp.shape) == 3:
        plt.imshow(cv2.cvtColor(image_with_kp, cv2.COLOR_BGR2RGB))
    else:  # Should be grayscale already if it reaches here and is 2D
        plt.imshow(image_with_kp, cmap="gray")

    plt.title(title)
    plt.axis("off")  # Turn off axis labels and ticks

    plt.savefig(
        output_path, bbox_inches="tight", dpi=300
    )  # Save with tight bounding box and high DPI
    print(f"Saved keypoints plot to {output_path}")

    if show_plot:
        plt.show()
    plt.close()  # Close the plot to free memory


def plot_and_save_matches(
    img1: np.ndarray,
    kp1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: List[cv2.KeyPoint],
    matches_to_draw: List[cv2.DMatch],
    output_path: str,
    title: str = "Feature Matches",
    show_plot: bool = False,
):
    """
    Draws matches between two images, saves the plot, and optionally displays it.

    Args:
        img1 (np.ndarray): The first input image (grayscale or color).
        kp1 (List[cv2.KeyPoint]): Keypoints from the first image.
        img2 (np.ndarray): The second input image (grayscale or color).
        kp2 (List[cv2.KeyPoint]): Keypoints from the second image.
        matches_to_draw (List[cv2.DMatch]): A list of DMatch objects representing
                                            the matches to be drawn.
        output_path (str): The full path including filename where the plot will be saved.
                           E.g., "plots/img1_img2_matches.png"
        title (str): The title for the plot.
        show_plot (bool): If True, displays the plot in a new window.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure images are in BGR format if they are grayscale for cv2.drawMatches,
    # as it expects 3-channel images for drawing colored lines.
    img1_display = img1
    img2_display = img2

    if len(img1.shape) == 2:  # Grayscale
        img1_display = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:  # Grayscale
        img2_display = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw matches
    img_matches_display = cv2.drawMatches(
        img1_display,
        kp1,
        img2_display,
        kp2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(16, 8))
    # cv2.drawMatches outputs a BGR image, convert to RGB for Matplotlib
    plt.imshow(cv2.cvtColor(img_matches_display, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

    plt.savefig(
        output_path, bbox_inches="tight", dpi=300
    )  # Save with tight bounding box and high DPI
    print(f"Saved matches plot to {output_path}")

    if show_plot:
        plt.show()
    plt.close()  # Close the plot to free memory


# --- Updated Visualization Function ---
def visualize_sfm_and_pose_open3d(
    points_3D: np.ndarray,
    camera_R_mats: Dict[int, np.ndarray],
    camera_t_vecs: Dict[int, np.ndarray],
    K_matrix: np.matrix,
    image_width: int,
    image_height: int,
    frustum_scale: float = 0.3,  # Scale factor for the camera frustum size
    point_colors: Optional[
        np.ndarray
    ] = None,  # Optional: N_points x 3 array of RGB colors (0-1)
    point_voxel_size: float = 0.0,  # Optional voxel downsampling size for point cloud.
    camera_color: Tuple[float, float, float] = (0.0, 0.8, 0.0),  # Green for cameras
):
    """
    Visualize the 3D sparse point cloud and camera poses using Open3D.

    Parameters:
    - points_3D: Nx3 NumPy array of 3D points.
    - camera_R_mats: Dictionary mapping image index to 3x3 rotation matrix.
    - camera_t_vecs: Dictionary mapping image index to 3x1 translation vector.
    - K_matrix: 3x3 camera intrinsic matrix.
    - image_width: Width of the images.
    - image_height: Height of the images.
    - frustum_scale: Controls the size of the visualized camera frustums.
    - point_colors: Optional Nx3 array for point cloud colors (RGB, 0-1 range).
    - camera_color: RGB tuple for camera frustum color (0-1 range).
    """
    if points_3D is None or points_3D.shape[0] == 0:
        print("No 3D points to visualize.")
        return

    # 1. Create Point Cloud Geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)

    if point_colors is not None and point_colors.shape[0] == points_3D.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        # Default color if not provided (e.g., gray)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([0.5, 0.5, 0.5]), (points_3D.shape[0], 1))
        )

    if point_voxel_size and point_voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(float(point_voxel_size))

    geometries_to_draw = [pcd]

    fx = K_matrix[0, 0]
    fy = K_matrix[1, 1]
    cx = K_matrix[0, 2]
    cy = K_matrix[1, 2]

    z_plane = frustum_scale  # Arbitrary depth for the frustum base
    x_ndc_corners = [
        (0 - cx) / fx,
        (image_width - cx) / fx,
        (image_width - cx) / fx,
        (0 - cx) / fx,
    ]
    y_ndc_corners = [
        (0 - cy) / fy,
        (0 - cy) / fy,
        (image_height - cy) / fy,
        (image_height - cy) / fy,
    ]

    # Points in camera's local coordinate system:
    # Point 0: Camera optical center (apex of the pyramid)
    cam_points_local = [np.array([0, 0, 0])]
    # Points 1-4: Corners of the frustum base
    for i in range(4):
        cam_points_local.append(
            np.array([x_ndc_corners[i] * z_plane, y_ndc_corners[i] * z_plane, z_plane])
        )

    cam_points_local = np.array(cam_points_local)

    # Lines for the pyramid: connect apex (0) to each base corner (1-4)
    # and connect base corners to form the square base (1-2, 2-3, 3-4, 4-1)
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],  # Apex to corners
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],  # Base edges
    ]

    for img_idx in camera_R_mats.keys():
        if img_idx not in camera_t_vecs:
            print(f"Warning: Missing translation for camera {img_idx}. Skipping.")
            continue
        R = camera_R_mats[img_idx]
        t = camera_t_vecs[img_idx].reshape(3, 1)

        # Camera rotation in world (orientation)
        R_cam_in_world = R.T
        # Camera position in world (center)
        t_cam_in_world = -R.T @ t

        frustum_points_world = []
        for p_local in cam_points_local:
            p_world = R_cam_in_world @ p_local.reshape(3, 1) + t_cam_in_world
            frustum_points_world.append(p_world.ravel())

        frustum_points_world = np.array(frustum_points_world)

        # Create LineSet for the frustum
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(frustum_points_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(
            [camera_color for _ in range(len(lines))]
        )

        geometries_to_draw.append(line_set)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries_to_draw,
        window_name="SfM Point Cloud and Cameras",
        point_show_normal=False,
        width=1000,
        height=800,
    )
