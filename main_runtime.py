import os
import sys
import glob
import numpy as np

from feature_correspondence import feature_correspondence_runtime as _fcr
from utils import (
    get_images,
    export_to_colmap,
    visualize_sfm_and_pose_open3d,
    compute_point_colors_for_visualization,
)

# Ensure main.py imports the runtime (writable) implementation without editing root-owned files.
sys.modules["feature_correspondence.feature_correspondence"] = _fcr

import main as _main


def _visualize_reconstruction_colored(
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
    x, y, z = [], [], []
    keep_indices = []
    for pt_idx, pt3 in enumerate(points_3d_with_views):
        if np.sum(np.abs(pt3.point3d[0])) < _main.VOXEL_VISUALIZATION_THRESHOLD:
            keep_indices.append(pt_idx)
            x.append(pt3.point3d[0][0])
            y.append(pt3.point3d[0][1])
            z.append(pt3.point3d[0][2])

    vpoints = np.array(list(zip(x, y, z)))
    if vpoints.shape[0] == 0:
        _main.logging.warning("No valid 3D points found for visualization after filtering.")
        return

    _main.logging.info("Reconstruction finished. Exporting to COLMAP format...")

    colmap_output_dir = os.path.join(base_path, "colmap_export", imgset)
    os.makedirs(colmap_output_dir, exist_ok=True)

    images_paths_for_export = sorted(
        glob.glob(
            os.path.join(base_path, "datasets", imgset, f"*.{img_type}"),
            recursive=True,
        )
    )
    images_color_data = get_images(
        base_path, imgset, img_type, len(images_paths_for_export), type_="color"
    )
    chosen_color_strategy = os.getenv("POINT_COLOR_STRATEGY", "average")

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

    point_colors_all = compute_point_colors_for_visualization(
        reconstructed_points3d_with_views=points_3d_with_views,
        loaded_images=images_color_data,
        all_keypoints=all_keypoints,
        reconstructed_R_mats=R_mats,
        image_height=img_h,
        image_width=img_w,
        point_color_strategy=chosen_color_strategy,
    )

    point_colors_viz = None
    if (
        point_colors_all.ndim == 2
        and point_colors_all.shape[1] == 3
        and point_colors_all.shape[0] == len(points_3d_with_views)
    ):
        point_colors_viz = point_colors_all[keep_indices]

    if os.getenv("SKIP_VIS") == "1":
        _main.logging.info("SKIP_VIS=1: COLMAP export completed, skipping Open3D window.")
        return

    visualize_sfm_and_pose_open3d(
        points_3D=vpoints,
        camera_R_mats=R_mats,
        camera_t_vecs=t_vecs,
        K_matrix=K,
        image_width=img_w,
        image_height=img_h,
        frustum_scale=0.3,
        point_colors=point_colors_viz,
    )
    _main.logging.info("3D visualization complete.")


_main.visualize_reconstruction = _visualize_reconstruction_colored

# Default runtime profile: faster matching, cleaner BA, and better loop closures.
os.environ.setdefault("XFEAT_USE_AMP", "1")
os.environ.setdefault("XFEAT_BATCH_SIZE", "8")
os.environ.setdefault("XFEAT_PAIR_WINDOW", "6")
os.environ.setdefault("XFEAT_GLOBAL_TOPK", "4")
os.environ.setdefault("XFEAT_GLOBAL_MIN_GAP", "8")
os.environ.setdefault("XFEAT_GLOBAL_MIN_SIM", "0.20")
os.environ.setdefault("XFEAT_MAX_MATCHES_PER_PAIR", "2500")
os.environ.setdefault("SIFT_PAIR_WINDOW", "6")
os.environ.setdefault("SIFT_GLOBAL_TOPK", "4")
os.environ.setdefault("SIFT_GLOBAL_MIN_GAP", "8")
os.environ.setdefault("SIFT_GLOBAL_MIN_SIM", "0.20")

os.environ.setdefault("BA_LOSS_TYPE", "huber")
os.environ.setdefault("BA_HUBER_DELTA", "1.0")
os.environ.setdefault("BA_EARLY_STOP_PATIENCE", "12")
os.environ.setdefault("BA_EARLY_STOP_TOL", "5e-4")
os.environ.setdefault("BA_MAX_POINTS_LOCAL", "6000")
os.environ.setdefault("BA_MAX_POINTS_GLOBAL", "12000")
os.environ.setdefault("BA_MIN_TRACK_LENGTH_LOCAL", "2")
os.environ.setdefault("BA_MIN_TRACK_LENGTH_GLOBAL", "3")
os.environ.setdefault("BA_PRUNE_OUTLIERS", "1")
os.environ.setdefault("BA_OUTLIER_THRESH", "3.0")
os.environ.setdefault("BA_MIN_TRACK_AFTER_PRUNE", "3")
os.environ.setdefault("BA_OPTIMIZE_INTRINSICS", "1")
os.environ.setdefault("BA_OPTIMIZE_INTRINSICS_MIN_CAMS", "24")
os.environ.setdefault("BA_INTRINSICS_REG", "5e-2")

os.environ.setdefault("TRI_MAX_REPROJ", "3.0")
os.environ.setdefault("TRI_MIN_ANGLE", "2.0")
os.environ.setdefault("PNP_SOLVER", "EPNP")
os.environ.setdefault("PNP_CONFIDENCE", "0.999")
os.environ.setdefault("PNP_MIN_INLIER_RATIO", "0.35")
os.environ.setdefault("PNP_MIN_INLIER_COUNT", "35")
os.environ.setdefault("PNP_MAX_REPROJ_ERR", "8.0")
os.environ.setdefault("PNP_TEST_REPROJ_THRESH", "5.0")

if os.getenv("FAST_MODE") == "1":
    # Feature extraction / matching
    os.environ.setdefault("XFEAT_TOP_K", "2048")
    os.environ.setdefault("XFEAT_PAIR_WINDOW", "4")
    os.environ.setdefault("XFEAT_GLOBAL_TOPK", "3")
    os.environ.setdefault("XFEAT_MAX_MATCHES_PER_PAIR", "3000")

    # Bundle adjustment speed/quality tradeoffs
    os.environ.setdefault("LOCAL_BA_N_ITER_TORCH", "200")
    os.environ.setdefault("GLOBAL_BA_N_ITER_TORCH", "500")
    os.environ.setdefault("LOCAL_BA_LEARNING_RATE_TORCH", "5e-4")
    os.environ.setdefault("GLOBAL_BA_LEARNING_RATE_TORCH", "5e-5")
    os.environ.setdefault("BA_EARLY_STOP_PATIENCE", "10")
    os.environ.setdefault("BA_EARLY_STOP_TOL", "5e-4")
    os.environ.setdefault("BA_MAX_POINTS_LOCAL", "4000")
    os.environ.setdefault("BA_MAX_POINTS_GLOBAL", "8000")
    os.environ.setdefault("BA_MIN_TRACK_LENGTH_LOCAL", "2")
    os.environ.setdefault("BA_MIN_TRACK_LENGTH_GLOBAL", "3")

    # PnP gating (faster + safer); relax if it rejects too much
    os.environ.setdefault("PNP_MIN_INLIER_RATIO", "0.35")
    os.environ.setdefault("PNP_MIN_INLIER_COUNT", "25")
    os.environ.setdefault("PNP_MAX_REPROJ_ERR", "12.0")

    # Skip visualization for speed in headless runs
    os.environ.setdefault("SKIP_VIS", "1")


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


# Override BA defaults without editing root-owned main.py.
_main.LOCAL_BA_N_ITER_TORCH = _env_int("LOCAL_BA_N_ITER_TORCH", 250)
_main.LOCAL_BA_LEARNING_RATE_TORCH = _env_float("LOCAL_BA_LEARNING_RATE_TORCH", 5e-4)
_main.GLOBAL_BA_N_ITER_TORCH = _env_int("GLOBAL_BA_N_ITER_TORCH", 650)
_main.GLOBAL_BA_LEARNING_RATE_TORCH = _env_float(
    "GLOBAL_BA_LEARNING_RATE_TORCH", 5e-5
)


if __name__ == "__main__":
    _main.main()
