import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from typing import Tuple, List
import os
import glob
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import MiniBatchKMeans # Efficient for large datasets

def get_images(base_path, dataset_path, img_format, use_n_imgs=-1, type_="color"):
    images_paths = sorted(
        glob.glob(
            os.path.join(base_path, "datasets", dataset_path) + "/*." + img_format,
            recursive=True,
        )
    )
    images = []
    if not use_n_imgs==-1 and use_n_imgs<=len(images_paths):
        images_paths=images_paths[:use_n_imgs]
        
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
    point_color_strategy: str = "first" # NEW: "first", "average", or "median"
):
    """
    Exports reconstruction data to COLMAP text format, with selectable RGB strategy.
    """
    os.makedirs(output_path, exist_ok=True)
    valid_color_strategies = ["first", "average", "median"]
    if point_color_strategy not in valid_color_strategies:
        raise ValueError(f"Invalid point_color_strategy. Choose from {valid_color_strategies}")

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

    colmap_img_id_map = {py_idx: col_idx for col_idx, py_idx in enumerate(sorted(reconstructed_R_mats.keys()), 1)}
    # --- 2. points3D.txt (modified for RGB strategy) ---
    points3d_lines_buffer = [] 
    point_rgb_colors = []

    num_valid_3d_points = 0
    track_lengths = []

    for pt3d_py_idx, pt_obj in enumerate(reconstructed_points3d_with_views):
        colmap_point3d_id = pt3d_py_idx + 1
        
        x_3d, y_3d, z_3d = pt_obj.point3d.ravel()
        
        r_val, g_val, b_val = 0, 0, 0 # Default color
        point_colors_bgr_list = [] # Store as list of (B,G,R) tuples

        sorted_observing_py_img_indices = sorted(pt_obj.source_2dpt_idxs.keys())

        for img_py_idx in sorted_observing_py_img_indices:
            if img_py_idx in reconstructed_R_mats:
                kpt_original_idx = pt_obj.source_2dpt_idxs[img_py_idx]
                
                if 0 <= img_py_idx < len(loaded_images) and \
                   0 <= kpt_original_idx < len(all_keypoints[img_py_idx]):
                    
                    kp = all_keypoints[img_py_idx][kpt_original_idx]
                    kp_x, kp_y = kp.pt
                    iy = int(round(kp_y))
                    ix = int(round(kp_x))

                    if 0 <= iy < image_height and 0 <= ix < image_width:
                        bgr_pixel = loaded_images[img_py_idx][iy, ix]
                        point_colors_bgr_list.append(bgr_pixel) # Store as (B, G, R)
        
        if point_colors_bgr_list:
            if point_color_strategy == "first":
                # Use the color from the first valid observation
                b_val, g_val, r_val = point_colors_bgr_list[0] 
            
            elif point_color_strategy == "average":
                # Average colors (convert to float for mean, then back to int)
                avg_b = int(round(np.mean([c[0] for c in point_colors_bgr_list])))
                avg_g = int(round(np.mean([c[1] for c in point_colors_bgr_list])))
                avg_r = int(round(np.mean([c[2] for c in point_colors_bgr_list])))
                b_val, g_val, r_val = avg_b, avg_g, avg_r
            
            elif point_color_strategy == "median":
                # Median colors (more robust to outliers)
                # np.median operates on flattened arrays or per-axis
                # For RGB, it's common to take median per channel
                median_b = int(round(np.median([c[0] for c in point_colors_bgr_list])))
                median_g = int(round(np.median([c[1] for c in point_colors_bgr_list])))
                median_r = int(round(np.median([c[2] for c in point_colors_bgr_list])))
                b_val, g_val, r_val = median_b, median_g, median_r
            
            # Ensure values are within 0-255 (though rounding from mean/median should be okay)
            r_val = np.clip(r_val, 0, 255)
            g_val = np.clip(g_val, 0, 255)
            b_val = np.clip(b_val, 0, 255)
        
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
            points3d_lines_buffer.append(f"{colmap_point3d_id} {x_3d} {y_3d} {z_3d} {final_r} {final_g} {final_b} {error} {track_str}\n")
            point_rgb_colors.append([final_r/255, final_g/255, final_b/255])
    with open(os.path.join(output_path, "points3D.txt"), "w") as f_pts3d:
        f_pts3d.write("# 3D point list with one line of data per point:\n")
        f_pts3d.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        mean_track_len_val = np.mean(track_lengths) if track_lengths else 0
        f_pts3d.write(f"# Number of points: {num_valid_3d_points}, mean track length: {mean_track_len_val}\n")
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

        images_lines_buffer.append(f"{colmap_image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_colmap_id} {img_name}\n")

        points2d_line_parts = []
        num_observations_in_image = 0
        
        img_kpt_to_pt3d_id = {}
        for pt3d_py_idx, pt_obj in enumerate(reconstructed_points3d_with_views):
            temp_track_len = 0
            for obs_img_idx in pt_obj.source_2dpt_idxs.keys():
                if obs_img_idx in reconstructed_R_mats:
                    temp_track_len +=1
            
            if temp_track_len >= 2:
                colmap_pt3d_id_current = pt3d_py_idx + 1
                if py_img_idx in pt_obj.source_2dpt_idxs:
                    kpt_orig_idx = pt_obj.source_2dpt_idxs[py_img_idx]
                    img_kpt_to_pt3d_id[(py_img_idx, kpt_orig_idx)] = colmap_pt3d_id_current
        
        for kpt_original_idx, kp in enumerate(all_keypoints[py_img_idx]):
            x_2d, y_2d = kp.pt
            observed_colmap_point3d_id = img_kpt_to_pt3d_id.get((py_img_idx, kpt_original_idx), -1)
            points2d_line_parts.extend([str(x_2d), str(y_2d), str(observed_colmap_point3d_id)])
            if observed_colmap_point3d_id != -1:
                num_observations_in_image += 1
        
        total_observations_for_header += num_observations_in_image
        images_lines_buffer.append(" ".join(points2d_line_parts) + "\n")

    with open(os.path.join(output_path, "images.txt"), "w") as f_img:
        f_img.write("# Image list with two lines of data per image:\n")
        f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_img.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        mean_obs_val = total_observations_for_header / len(colmap_img_id_map) if colmap_img_id_map else 0
        f_img.write(f"# Number of images: {len(colmap_img_id_map)}, mean observations per image: {mean_obs_val}\n")
        for line in images_lines_buffer:
            f_img.write(line)

    print(f"COLMAP data exported to {output_path} using '{point_color_strategy}' color strategy.")
    return np.array(point_rgb_colors)

def plot_and_save_keypoints(image: np.ndarray, keypoints: List[cv2.KeyPoint],
                            output_path: str, title: str = "Detected Keypoints",
                            show_plot: bool = False):
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
        display_image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(10, 8))
    # Convert BGR (OpenCV) to RGB (Matplotlib) if the image is color
    if len(image_with_kp.shape) == 3:
        plt.imshow(cv2.cvtColor(image_with_kp, cv2.COLOR_BGR2RGB))
    else: # Should be grayscale already if it reaches here and is 2D
        plt.imshow(image_with_kp, cmap='gray')

    plt.title(title)
    plt.axis("off") # Turn off axis labels and ticks

    plt.savefig(output_path, bbox_inches='tight', dpi=300) # Save with tight bounding box and high DPI
    print(f"Saved keypoints plot to {output_path}")

    if show_plot:
        plt.show()
    plt.close() # Close the plot to free memory


def plot_and_save_matches(img1: np.ndarray, kp1: List[cv2.KeyPoint],
                          img2: np.ndarray, kp2: List[cv2.KeyPoint],
                          matches_to_draw: List[cv2.DMatch],
                          output_path: str, title: str = "Feature Matches",
                          show_plot: bool = False):
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

    if len(img1.shape) == 2: # Grayscale
        img1_display = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2: # Grayscale
        img2_display = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw matches
    img_matches_display = cv2.drawMatches(
        img1_display, kp1, img2_display, kp2, matches_to_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(16, 8))
    # cv2.drawMatches outputs a BGR image, convert to RGB for Matplotlib
    plt.imshow(cv2.cvtColor(img_matches_display, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

    plt.savefig(output_path, bbox_inches='tight', dpi=300) # Save with tight bounding box and high DPI
    print(f"Saved matches plot to {output_path}")

    if show_plot:
        plt.show()
    plt.close() # Close the plot to free memory


# --- Updated Visualization Function ---
def visualize_sfm_and_pose_open3d(
    points_3D: np.ndarray,
    camera_R_mats: Dict[int, np.ndarray],
    camera_t_vecs: Dict[int, np.ndarray],
    K_matrix: np.matrix,
    image_width: int,
    image_height: int,
    frustum_scale: float = 0.3,  # Scale factor for the camera frustum size
    point_colors: Optional[np.ndarray] = None, # Optional: N_points x 3 array of RGB colors (0-1)
    camera_color: Tuple[float, float, float] = (0.0, 0.8, 0.0) # Green for cameras
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

    geometries_to_draw = [pcd]

    fx = K_matrix[0, 0]
    fy = K_matrix[1, 1]
    cx = K_matrix[0, 2]
    cy = K_matrix[1, 2]

    z_plane = frustum_scale # Arbitrary depth for the frustum base
    x_ndc_corners = [(0 - cx) / fx, (image_width - cx) / fx, (image_width - cx) / fx, (0 - cx) / fx]
    y_ndc_corners = [(0 - cy) / fy, (0 - cy) / fy, (image_height - cy) / fy, (image_height - cy) / fy]

    # Points in camera's local coordinate system:
    # Point 0: Camera optical center (apex of the pyramid)
    cam_points_local = [np.array([0, 0, 0])]
    # Points 1-4: Corners of the frustum base
    for i in range(4):
        cam_points_local.append(np.array([x_ndc_corners[i] * z_plane, y_ndc_corners[i] * z_plane, z_plane]))

    cam_points_local = np.array(cam_points_local)

    # Lines for the pyramid: connect apex (0) to each base corner (1-4)
    # and connect base corners to form the square base (1-2, 2-3, 3-4, 4-1)
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Apex to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Base edges
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
            p_world = R_cam_in_world @ p_local.reshape(3,1) + t_cam_in_world
            frustum_points_world.append(p_world.ravel())
        
        frustum_points_world = np.array(frustum_points_world)

        # Create LineSet for the frustum
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(frustum_points_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([camera_color for _ in range(len(lines))])
        
        geometries_to_draw.append(line_set)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries_to_draw,
        window_name="SfM Point Cloud and Cameras",
        point_show_normal=False,
        width=1000,
        height=800
    )

class BoVW:
    def __init__(self, vocabulary_size=1000, max_descriptors_for_vocab_build=500000, random_state=42):
        """
        Initializes the Bag of Visual Words handler.
        Args:
            vocabulary_size (int): The desired number of visual words in the vocabulary.
            max_descriptors_for_vocab_build (int): Maximum number of descriptors to use for building vocabulary
                                                   to keep memory and time manageable.
            random_state (int): Random seed for reproducibility of clustering.
        """
        self.vocabulary_size = vocabulary_size
        self.max_descriptors_for_vocab_build = max_descriptors_for_vocab_build
        self.random_state = random_state
        self.vocabulary = None  # This will store the k-means cluster centers (visual words)
        self.bovw_database = {} # Stores BoVW histograms for processed images: {img_idx: histogram}
        self._descriptors_for_vocab_training = [] # Temporarily stores descriptors for training

    def add_descriptors_for_training(self, list_of_descriptor_arrays):
        """
        Collects descriptors from images to be used for vocabulary training.
        Call this for all images before calling build_vocabulary.
        Args:
            list_of_descriptor_arrays (list): A list where each element is a NumPy array of descriptors
                                             (e.g., SIFT descriptors) for an image.
        """
        if self.vocabulary is not None:
            print("Warning: Vocabulary already built. Not adding more descriptors for training.")
            return

        for desc_array in list_of_descriptor_arrays:
            if desc_array is not None and len(desc_array) > 0:
                self._descriptors_for_vocab_training.extend(desc_array)

        current_pool_size = len(self._descriptors_for_vocab_training)
        if current_pool_size > self.max_descriptors_for_vocab_build:
            print(f"Descriptor pool for vocab training ({current_pool_size}) exceeds limit ({self.max_descriptors_for_vocab_build}). Subsampling.")
            indices = np.random.choice(current_pool_size, self.max_descriptors_for_vocab_build, replace=False)
            self._descriptors_for_vocab_training = [self._descriptors_for_vocab_training[i] for i in indices]

    def build_vocabulary(self):
        """
        Builds the visual vocabulary using k-means clustering on the collected descriptors.
        """
        if self.vocabulary is not None:
            print("Vocabulary already built.")
            return

        if not self._descriptors_for_vocab_training:
            print("Error: No descriptors collected for vocabulary training.")
            return

        descriptors_for_training = np.array(self._descriptors_for_vocab_training)
        actual_num_descriptors = descriptors_for_training.shape[0]

        if actual_num_descriptors == 0:
            print("Error: Descriptor training array is empty after conversion.")
            self._descriptors_for_vocab_training = [] # Clear temp storage
            return

        print(f"Building vocabulary with {actual_num_descriptors} descriptors and target vocab size {self.vocabulary_size}...")

        if actual_num_descriptors < self.vocabulary_size:
            print(f"Warning: Number of unique descriptors ({actual_num_descriptors}) is less than "
                  f"target vocabulary size ({self.vocabulary_size}). Using all descriptors as vocabulary.")
            self.vocabulary = np.unique(descriptors_for_training, axis=0)
            self.vocabulary_size = self.vocabulary.shape[0]
        else:
            # Using MiniBatchKMeans for efficiency, especially with many descriptors
            kmeans = MiniBatchKMeans(n_clusters=self.vocabulary_size,
                                     random_state=self.random_state,
                                     batch_size=min(self.vocabulary_size * 5, actual_num_descriptors // 5 + 1), # Ensure batch_size is reasonable
                                     n_init='auto',
                                     max_iter=100) # Can tune max_iter
            kmeans.fit(descriptors_for_training)
            self.vocabulary = kmeans.cluster_centers_

        print(f"Vocabulary built with {self.vocabulary.shape[0]} visual words.")
        self._descriptors_for_vocab_training = [] # Clear temporary storage to save memory

    def descriptors_to_bovw_histogram(self, image_descriptors):
        """
        Converts a set of descriptors for a single image to a BoVW histogram.
        Args:
            image_descriptors (np.ndarray): NumPy array of descriptors for the image.
        Returns:
            np.ndarray: A normalized BoVW histogram (L1 norm). Returns zeros if no vocab or descriptors.
        """
        if self.vocabulary is None:
            # print("Warning: Vocabulary not built yet. Cannot create BoVW histogram.")
            return np.zeros(self.vocabulary_size if self.vocabulary_size > 0 else 1) # Avoid zero-size array
        if image_descriptors is None or len(image_descriptors) == 0:
            return np.zeros(self.vocabulary_size)

        # For SIFT descriptors, NORM_L2 is standard.
        # BFMatcher can be used to find the nearest visual word for each descriptor.
        # Ensure image_descriptors are float32 as SIFT descriptors are
        if image_descriptors.dtype != np.float32:
            image_descriptors = image_descriptors.astype(np.float32)
        if self.vocabulary.dtype != np.float32: # Vocabulary should also be float32
            self.vocabulary = self.vocabulary.astype(np.float32)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        # Query: image_descriptors, Train: vocabulary (the visual words)
        matches = bf.match(queryDescriptors=image_descriptors, trainDescriptors=self.vocabulary)
        
        word_indices = [m.trainIdx for m in matches if m.trainIdx < self.vocabulary_size] # Safety check

        histogram = np.zeros(self.vocabulary_size)
        for idx in word_indices:
            histogram[idx] += 1

        # Normalize the histogram (L1 norm)
        sum_hist = np.sum(histogram)
        if sum_hist > 0:
            histogram = histogram / sum_hist
        return histogram

    def add_image_to_bovw_db(self, img_idx, image_descriptors):
        """
        Computes the BoVW histogram for an image and adds/updates it in the database.
        Args:
            img_idx (int): The index of the image.
            image_descriptors (np.ndarray): Descriptors for the image.
        """
        if self.vocabulary is None:
            # This warning can be frequent if vocab building is deferred or fails.
            # Consider logging it less verbosely in a real application.
            # print(f"Debug: Vocab not ready, cannot add img {img_idx} to BoVW DB.")
            return
        histogram = self.descriptors_to_bovw_histogram(image_descriptors)
        self.bovw_database[img_idx] = histogram

    def query_similar_images(self, query_img_idx, query_descriptors, top_n=5, skip_recent_frames=5, min_similarity_score=0.05):
        """
        Finds the most similar images in the database to the query image,
        excluding recent frames and the query image itself.
        Args:
            query_img_idx (int): Index of the current query image.
            query_descriptors (np.ndarray): Descriptors of the current query image.
            top_n (int): Maximum number of similar images to return.
            skip_recent_frames (int): Number of chronologically adjacent frames to ignore.
            min_similarity_score (float): Minimum similarity score to consider a match.
        Returns:
            list: A list of tuples (similarity_score, image_index), sorted by score.
        """
        if self.vocabulary is None or not self.bovw_database:
            return []

        query_histogram = self.descriptors_to_bovw_histogram(query_descriptors)
        if np.sum(query_histogram) == 0: # No features in query image or other issue
            return []

        similarities = []
        for db_img_idx, db_histogram in self.bovw_database.items():
            # Avoid comparing with itself or very recent/adjacent images to find meaningful non-sequential loops
            if db_img_idx == query_img_idx or abs(query_img_idx - db_img_idx) <= skip_recent_frames:
                continue

            # Using Cosine Similarity: (A dot B) / (||A|| * ||B||)
            # For L1 normalized histograms, this is equivalent to dot product if histograms are treated as unit vectors,
            # but it's safer to compute norms explicitly.
            dot_product = np.dot(query_histogram, db_histogram)
            norm_query = np.linalg.norm(query_histogram)
            norm_db = np.linalg.norm(db_histogram)

            similarity = 0.0
            if norm_query > 1e-6 and norm_db > 1e-6: # Avoid division by zero
                similarity = dot_product / (norm_query * norm_db)
            
            if similarity >= min_similarity_score:
                similarities.append((similarity, db_img_idx))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_n]
