import numpy as np
import os
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import torch
import torch.optim as optim


def create_bundle_adjustment_sparsity(
    n_cameras, n_points, camera_indices, point_indices
):
    """
    Creates the sparsity matrix for bundle adjustment.

    :param n_cameras: Number of cameras.
    :param n_points: Number of 3D points.
    :param camera_indices: Array of camera indices for each 2D point observation.
    :param point_indices: Array of 3D point indices for each 2D point observation.
    :return: Sparse matrix representing the Jacobian sparsity pattern.
    """
    m = camera_indices.size * 2
    n = n_cameras * 12 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    row_indices = np.arange(camera_indices.size)
    for s in range(12):
        A[2 * row_indices, camera_indices * 12 + s] = 1
        A[2 * row_indices + 1, camera_indices * 12 + s] = 1

    for s in range(3):
        A[2 * row_indices, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * row_indices + 1, n_cameras * 12 + point_indices * 3 + s] = 1

    return A


def project_points(points_3d, camera_params, K):
    """
    Projects 3D points onto the image plane using given camera parameters and intrinsics.

    :param points_3d: (N, 3) array of 3D point coordinates.
    :param camera_params: (M, 12) array of camera parameters (rotation and translation).
    :param K: (3, 3) camera intrinsics matrix.
    :return: List of (num_points_in_camera, 2) projected 2D point coordinates.
    """
    projected_points = []
    for cam_param, point in zip(camera_params, points_3d):
        R = cam_param[:9].reshape(3, 3)
        rvec, _ = cv2.Rodrigues(R)
        t = cam_param[9:].reshape(3, 1)
        point = np.expand_dims(point, axis=0)
        projected, _ = cv2.projectPoints(point, rvec, t, K, distCoeffs=np.array([]))
        projected_points.append(np.squeeze(projected))
    return projected_points


def calculate_reprojection_error(
    params, n_cameras, n_points, camera_indices, point_indices, points_2d, K, cuda=False
):
    """Calculates the reprojection error for a given set of camera and 3D point parameters.

    This function takes the current camera and 3D point parameters, projects the 3D points
    onto the 2D image planes using the specified camera parameters and intrinsic matrix,
    and then computes the difference between these projected points and the observed
    2D points. This difference, or reprojection error, is a common metric used in
    bundle adjustment to optimize camera poses and 3D point locations.

    Args:
        params (np.ndarray or torch.Tensor): A 1D array containing all optimization parameters.
                                             The first `n_cameras * 12` elements represent
                                             camera parameters (12 per camera, typically
                                             3 for rotation vector, 3 for translation vector,
                                             and 6 for radial and tangential distortion
                                             coefficients, though the exact 12 depend on the
                                             `project_points` implementation). The remaining
                                             `n_points * 3` elements represent the 3D
                                             coordinates of the points.
        n_cameras (int): The total number of cameras.
        n_points (int): The total number of 3D points.
        camera_indices (np.ndarray): An array of shape (N,) where N is the number of
                                     observations. Each element indicates the index of
                                     the camera that observed a particular 2D point.
        point_indices (np.ndarray): An array of shape (N,) where N is the number of
                                    observations. Each element indicates the index of
                                    the 3D point corresponding to a particular 2D point.
        points_2d (np.ndarray): An array of shape (N, 2) containing the observed 2D
                                coordinates of the points in the image planes.
        K (np.ndarray): The camera intrinsic matrix of shape (3, 3).
        cuda (bool, optional): If True, the output reprojection error will be a
                               PyTorch tensor on the CUDA device. If False, it
                               will be a NumPy array. Defaults to False.

    Returns:
        np.ndarray or torch.Tensor: A 1D array or PyTorch tensor containing the flattened
                                    reprojection errors for all observed points. The
                                    shape will be (N * 2,), where N is the number of
                                    observations.

    Raises:
        ImportError: If `torch` is not installed and `cuda` is True.
        NameError: If `project_points` function is not defined in the scope.
                   (Note: `project_points` is a dependency and should be defined
                   elsewhere in the code or imported.)
    """
    # Reshape the input parameters into camera parameters and 3D points
    # Each camera has 12 parameters (e.g., rotation, translation, distortion)
    camera_params = params[: n_cameras * 12].reshape((n_cameras, 12))
    # Each 3D point has 3 coordinates (x, y, z)
    points_3d = params[n_cameras * 12 :].reshape((n_points, 3))

    # Project the 3D points onto the 2D image planes using the specified camera and point indices
    # and the intrinsic matrix K.
    # The `project_points` function is assumed to be defined elsewhere and handle the projection logic.
    projected_points = project_points(
        points_3d[point_indices], camera_params[camera_indices], K
    )

    # Calculate the reprojection error (difference between projected and observed 2D points)
    if not cuda:
        # If CUDA is not requested, return a NumPy array
        return (np.array(projected_points) - points_2d).ravel()
    else:
        # If CUDA is requested, convert the error to a PyTorch tensor and move it to the CUDA device
        # Ensure that `projected_points` is also handled appropriately (e.g., if it's a torch tensor already)
        # Note: The original code snippet was missing `return` here, which has been added.
        return torch.tensor(
            (np.array(projected_points) - points_2d).ravel(), device="cuda"
        )


def do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol):
    """
    Performs bundle adjustment to refine camera poses and 3D point coordinates.

    :param points3d_with_views: List of Point3D_with_views objects.
    :param R_mats: Dictionary mapping resected camera indices to their rotation matrices.
    :param t_vecs: Dictionary mapping resected camera indices to their translation vectors.
    :param resected_imgs: List of indices of resected images.
    :param keypoints: List of lists of cv2.Keypoint objects for each image.
    :param K: (3, 3) camera intrinsics matrix.
    :param ftol: Tolerance for change in the cost function for optimization.
    :return: Tuple containing updated points3d_with_views, R_mats, and t_vecs.
    """
    point_indices_list = []
    points_2d_list = []
    camera_indices_list = []
    points_3d_list = []
    initial_camera_params = []
    camera_index_map = {}
    camera_count = 0

    for img_index in resected_imgs:
        camera_index_map[img_index] = camera_count
        initial_camera_params.append(
            np.hstack((R_mats[img_index].ravel(), t_vecs[img_index].ravel()))
        )
        camera_count += 1

    for pt3d_idx, pt3d_with_view in enumerate(points3d_with_views):
        points_3d_list.append(pt3d_with_view.point3d.flatten())
        for cam_idx, kpt_idx in pt3d_with_view.source_2dpt_idxs.items():
            if cam_idx not in resected_imgs:
                continue
            point_indices_list.append(pt3d_idx)
            camera_indices_list.append(camera_index_map[cam_idx])
            points_2d_list.append(keypoints[cam_idx][kpt_idx].pt)

    if not points_3d_list:
        print("Warning: No common observations found for bundle adjustment.")
        return points3d_with_views, R_mats, t_vecs

    point_indices = np.array(point_indices_list)
    points_2d = np.array(points_2d_list)
    camera_indices = np.array(camera_indices_list)
    initial_points_3d = np.array(points_3d_list)
    initial_camera_params = np.array(initial_camera_params)

    n_cameras = initial_camera_params.shape[0]
    n_points = initial_points_3d.shape[0]
    initial_params = np.hstack(
        (initial_camera_params.ravel(), initial_points_3d.ravel())
    )
    sparsity_matrix = create_bundle_adjustment_sparsity(
        n_cameras, n_points, camera_indices, point_indices
    )

    optimization_result = least_squares(
        calculate_reprojection_error,
        initial_params,
        jac_sparsity=sparsity_matrix,
        verbose=0,
        x_scale="jac",
        loss="huber",
        ftol=ftol,
        xtol=1e-4,
        gtol=1e-4,
        method="trf",
        max_nfev=2000,
        f_scale=1.0,
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
    )

    adjusted_camera_params = optimization_result.x[: n_cameras * 12].reshape(
        n_cameras, 12
    )
    adjusted_points_3d = optimization_result.x[n_cameras * 12 :].reshape(n_points, 3)
    updated_R_mats = {}
    updated_t_vecs = {}

    for true_index, normalized_index in camera_index_map.items():
        updated_R_mats[true_index] = adjusted_camera_params[normalized_index][
            :9
        ].reshape(3, 3)
        updated_t_vecs[true_index] = adjusted_camera_params[normalized_index][
            9:
        ].reshape(3, 1)

    for i, pt3d_with_view in enumerate(points3d_with_views):
        pt3d_with_view.point3d = adjusted_points_3d[i].reshape(1, 3)

    return points3d_with_views, updated_R_mats, updated_t_vecs


def do_BA_cuda(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol):
    """
    Performs bundle adjustment to refine camera poses and 3D point coordinates.

    :param points3d_with_views: List of Point3D_with_views objects.
    :param R_mats: Dictionary mapping resected camera indices to their rotation matrices.
    :param t_vecs: Dictionary mapping resected camera indices to their translation vectors.
    :param resected_imgs: List of indices of resected images.
    :param keypoints: List of lists of cv2.Keypoint objects for each image.
    :param K: (3, 3) camera intrinsics matrix.
    :param ftol: Tolerance for change in the cost function for optimization.
    :return: Tuple containing updated points3d_with_views, R_mats, and t_vecs.

    :description:
        # CUDA implementation of bundle adjustment
        # This is a simplified version of the original code
        # It assumes that the input data is already preprocessed and ready for BA
        # It also assumes that the camera intrinsics are already known
        # The code is written in a way that it can be easily extended to support more features
        # For example, it can be easily modified to support multiple views per point or multiple points per
        # view, or to support different types of camera models, etc.
        # The code is also written in a way that it can be easily parallelized using CUDA
        # For example, it can be easily modified to use multiple threads to process different points or views
        # simultaneously
        # The code is also written in a way that it can be easily extended to support different types
        # of optimization algorithms, such as Gauss-Newton or Levenberg-Marquardt
        # The code is also written in a way that it can be easily modified to support different types
        # of cost functions, such as the standard reprojection error or a more advanced cost function
        # that takes into account the uncertainty of the measurements
        # The code is also written in a way that it can be easily extended to support different types
        # of regularization terms, such as the standard L2 regularization or a more advanced regularization
        # term that takes into account the structure of the problem
    """

    point_indices_list = []
    points_2d_list = []
    camera_indices_list = []
    points_3d_list = []
    initial_camera_params = []
    camera_index_map = {}
    camera_count = 0

    for img_index in resected_imgs:
        camera_index_map[img_index] = camera_count
        initial_camera_params.append(
            np.hstack((R_mats[img_index].ravel(), t_vecs[img_index].ravel()))
        )
        camera_count += 1

    for pt3d_idx, pt3d_with_view in enumerate(points3d_with_views):
        points_3d_list.append(pt3d_with_view.point3d.flatten())
        for cam_idx, kpt_idx in pt3d_with_view.source_2dpt_idxs.items():
            if cam_idx not in resected_imgs:
                continue
            point_indices_list.append(pt3d_idx)
            camera_indices_list.append(camera_index_map[cam_idx])
            points_2d_list.append(keypoints[cam_idx][kpt_idx].pt)

    if not points_3d_list:
        print("Warning: No common observations found for bundle adjustment.")
        return points3d_with_views, R_mats, t_vecs

    point_indices = np.array(point_indices_list)
    points_2d = np.array(points_2d_list)
    camera_indices = np.array(camera_indices_list)
    initial_points_3d = np.array(points_3d_list)
    initial_camera_params = np.array(initial_camera_params)

    n_cameras = initial_camera_params.shape[0]
    n_points = initial_points_3d.shape[0]
    initial_params = np.hstack(
        (initial_camera_params.ravel(), initial_points_3d.ravel())
    )
    sparsity_matrix = create_bundle_adjustment_sparsity(
        n_cameras, n_points, camera_indices, point_indices
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Optimizer setup
    params = torch.tensor(
        initial_params, dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = optim.LBFGS([params], lr=1, max_iter=50, history_size=10)

    # Define closure for LBFGS
    def closure():
        optimizer.zero_grad()
        reprojection_error = calculate_reprojection_error(
            params,
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
            points_2d,
            K,
            cuda=True,
        )
        loss = reprojection_error.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    optimization_result = least_squares(
        calculate_reprojection_error,
        initial_params,
        jac_sparsity=sparsity_matrix,
        verbose=2,
        x_scale="jac",
        loss="linear",
        ftol=ftol,
        xtol=1e-12,
        method="trf",
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
    )

    adjusted_camera_params = optimization_result.x[: n_cameras * 12].reshape(
        n_cameras, 12
    )
    adjusted_points_3d = optimization_result.x[n_cameras * 12 :].reshape(n_points, 3)
    updated_R_mats = {}
    updated_t_vecs = {}

    for true_index, normalized_index in camera_index_map.items():
        updated_R_mats[true_index] = adjusted_camera_params[normalized_index][
            :9
        ].reshape(3, 3)
        updated_t_vecs[true_index] = adjusted_camera_params[normalized_index][
            9:
        ].reshape(3, 1)

    for i, pt3d_with_view in enumerate(points3d_with_views):
        pt3d_with_view.point3d = adjusted_points_3d[i].reshape(1, 3)

    return points3d_with_views, updated_R_mats, updated_t_vecs


def project_points_torch(points_3d_obs, R_matrices_obs, t_vectors_obs, K_tensor):
    """
    Projects 3D points to 2D using PyTorch operations.
    points_3d_obs: Tensor of shape (N_obs, 3), 3D points for each observation.
    R_matrices_obs: Tensor of shape (N_obs, 3, 3), rotation matrix for each observation.
    t_vectors_obs: Tensor of shape (N_obs, 3, 1), translation vector for each observation.
    K_tensor: Tensor of shape (3, 3), camera intrinsic matrix.
    Returns: Tensor of shape (N_obs, 2), projected 2D points.
    """
    N_obs = points_3d_obs.shape[0]

    # Transform to camera coordinates: P_cam = R @ P_world + t
    # R_matrices_obs @ points_3d_obs.unsqueeze(2) -> (N_obs, 3, 1)
    points_cam = R_matrices_obs @ points_3d_obs.unsqueeze(2) + t_vectors_obs

    # Project to image plane: p_img_homogeneous = K @ P_cam
    # K_tensor @ points_cam -> (N_obs, 3, 1)
    points_homogeneous_img = K_tensor @ points_cam

    # Normalize by the z-coordinate (perspective division)
    # Add a small epsilon to prevent division by zero and handle points behind camera.
    # A more robust solution would involve checking z > 0.
    epsilon = 1e-8
    z_coords = points_homogeneous_img[:, 2, 0]

    # Clamp z_coords to avoid negative or very small z issues leading to NaNs or Infs
    # This is a practical heuristic; proper handling involves checking visibility.
    z_coords_clamped = torch.clamp(z_coords, min=epsilon)

    projected_u = points_homogeneous_img[:, 0, 0] / z_coords_clamped
    projected_v = points_homogeneous_img[:, 1, 0] / z_coords_clamped

    projected_2d = torch.stack((projected_u, projected_v), dim=1)  # Shape: (N_obs, 2)
    return projected_2d


def calculate_reprojection_error_torch(
    params_tensor,
    n_cameras_optim,
    n_points_optim,
    camera_indices_tensor,
    point_indices_tensor,
    points_2d_tensor,
    K_tensor,
):
    """
    Calculates the reprojection error for bundle adjustment using PyTorch.
    params_tensor: 1D tensor. Contains n_cameras_optim*12 camera parameters (9 for R, 3 for t)
                   followed by n_points_optim*3 for 3D point coordinates.
    """
    num_cam_params_per_cam = 12  # 9 for R (3x3 matrix) + 3 for t

    # Extract camera parameters and 3D points from the flat params_tensor
    camera_params_flat = params_tensor[: n_cameras_optim * num_cam_params_per_cam]
    points_3d_flat = params_tensor[n_cameras_optim * num_cam_params_per_cam :]

    # Reshape to get per-camera parameters and per-point coordinates
    all_camera_params_optim = camera_params_flat.reshape(
        n_cameras_optim, num_cam_params_per_cam
    )
    all_points_3d_optim = points_3d_flat.reshape(n_points_optim, 3)

    # Select the specific camera parameters and 3D points for each observation
    # using the mapped indices
    observed_camera_params = all_camera_params_optim[camera_indices_tensor]
    observed_points_3d = all_points_3d_optim[point_indices_tensor]

    # Extract R (rotation matrices) and t (translation vectors) for each observation
    R_matrices_obs = observed_camera_params[:, :9].reshape(-1, 3, 3)
    t_vectors_obs = observed_camera_params[:, 9:].reshape(
        -1, 3, 1
    )  # Ensure it's a column vector

    # Project points
    projected_points_obs = project_points_torch(
        observed_points_3d, R_matrices_obs, t_vectors_obs, K_tensor
    )

    # Calculate error
    error = (
        projected_points_obs - points_2d_tensor
    ).ravel()  # Flatten to 1D for loss calculation
    return error


# --- Main PyTorch Bundle Adjustment Function ---


def do_BA_pytorch(
    points3d_with_views,
    R_mats,
    t_vecs,
    resected_imgs_orig_indices,
    all_keypoints,
    K_np,
    n_iterations=100,
    learning_rate=1e-3,
    apply_orthogonalization=True,
    loss_type="huber",
    huber_delta=1.0,
    early_stop_patience=25,
    early_stop_tol=1e-4,
    max_points=None,
    min_track_length=2,
    device_override=None,
):
    """
    Performs bundle adjustment using PyTorch on GPU if available.

    Args:
        points3d_with_views (list): List of Point3DWithViews objects.
        R_mats (dict): Dict mapping original image index to its 3x3 rotation matrix (np.ndarray).
        t_vecs (dict): Dict mapping original image index to its 3x1 translation vector (np.ndarray).
        resected_imgs_orig_indices (list): List of original indices of images currently resected and active in BA.
        all_keypoints (list): List of lists of cv2.KeyPoint objects for all images.
        K_np (np.matrix or np.ndarray): Camera intrinsic matrix (3x3).
        n_iterations (int): Number of optimization iterations.
        learning_rate (float): Learning rate for the Adam optimizer.
        apply_orthogonalization (bool): Whether to apply SVD-based orthogonalization to R matrices after optimization.

    Returns:
        Updated points3d_with_views, R_mats, t_vecs.
    """

    # Allow environment overrides without touching callers.
    loss_type = os.getenv("BA_LOSS_TYPE", loss_type)
    huber_delta = float(os.getenv("BA_HUBER_DELTA", huber_delta))
    early_stop_patience = int(os.getenv("BA_EARLY_STOP_PATIENCE", early_stop_patience))
    early_stop_tol = float(os.getenv("BA_EARLY_STOP_TOL", early_stop_tol))
    max_points_env = os.getenv("BA_MAX_POINTS")
    if max_points is None and max_points_env:
        try:
            max_points = int(max_points_env)
        except ValueError:
            max_points = None
    min_track_length_env = os.getenv("BA_MIN_TRACK_LENGTH")
    if min_track_length is None and min_track_length_env:
        try:
            min_track_length = int(min_track_length_env)
        except ValueError:
            min_track_length = 2
    if device_override is None:
        device_override = os.getenv("BA_DEVICE")

    print(f"Starting PyTorch BA for {len(resected_imgs_orig_indices)} cameras.")

    # 1. Prepare data and mappings
    # Map original camera indices to new dense indices (0 to n_cameras_optim-1) for optimization
    cam_orig_to_optim_idx = {
        orig_idx: i
        for i, orig_idx in enumerate(sorted(list(resected_imgs_orig_indices)))
    }
    n_cameras_optim = len(cam_orig_to_optim_idx)

    # Local/global overrides based on active camera count
    global_min_cams = int(os.getenv("BA_GLOBAL_MIN_CAMERAS", "10"))
    if n_cameras_optim >= global_min_cams:
        max_points_global = os.getenv("BA_MAX_POINTS_GLOBAL")
        if max_points_global:
            try:
                max_points = int(max_points_global)
            except ValueError:
                pass
        min_track_global = os.getenv("BA_MIN_TRACK_LENGTH_GLOBAL")
        if min_track_global:
            try:
                min_track_length = int(min_track_global)
            except ValueError:
                pass
    else:
        max_points_local = os.getenv("BA_MAX_POINTS_LOCAL")
        if max_points_local:
            try:
                max_points = int(max_points_local)
            except ValueError:
                pass
        min_track_local = os.getenv("BA_MIN_TRACK_LENGTH_LOCAL")
        if min_track_local:
            try:
                min_track_length = int(min_track_local)
            except ValueError:
                pass

    if min_track_length is None:
        min_track_length = 2

    initial_camera_params_list = [None] * n_cameras_optim
    for orig_idx, optim_idx in cam_orig_to_optim_idx.items():
        if orig_idx in R_mats and orig_idx in t_vecs:
            initial_camera_params_list[optim_idx] = np.hstack(
                (R_mats[orig_idx].ravel(), t_vecs[orig_idx].ravel())
            )
        else:
            print(
                f"Warning: Camera original index {orig_idx} not found in R_mats/t_vecs during BA prep."
            )
            # Handle this case: skip BA or raise error
            return points3d_with_views, R_mats, t_vecs

    if any(p is None for p in initial_camera_params_list):
        print("Error: Could not initialize all camera parameters for BA.")
        return points3d_with_views, R_mats, t_vecs

    initial_camera_params_np = np.array(initial_camera_params_list, dtype=np.float32)

    # Prepare 3D points and observations
    # Map original Point3DWithViews object's list index to new dense point index (0 to n_points_optim-1)
    pt3d_orig_list_idx_to_optim_idx = {}
    optim_idx_to_pt3d_orig_list_idx = {}

    current_optim_pt_idx = 0

    obs_cam_optim_indices = []
    obs_pt_optim_indices = []
    obs_points_2d_list = []
    initial_points_3d_optim_list = []

    # Build candidate points with active observations
    candidates = []
    for i_orig_list_idx, pt3d_obj in enumerate(points3d_with_views):
        active_views = []
        for cam_orig_idx_viewing_pt in pt3d_obj.source_2dpt_idxs.keys():
            if cam_orig_idx_viewing_pt in cam_orig_to_optim_idx:
                active_views.append(cam_orig_idx_viewing_pt)

        if len(active_views) >= min_track_length:
            candidates.append((len(active_views), i_orig_list_idx, active_views))

    if max_points is not None and max_points > 0 and len(candidates) > max_points:
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_points]

    for _, i_orig_list_idx, active_views in candidates:
        pt3d_obj = points3d_with_views[i_orig_list_idx]
        pt3d_orig_list_idx_to_optim_idx[i_orig_list_idx] = current_optim_pt_idx
        optim_idx_to_pt3d_orig_list_idx[current_optim_pt_idx] = i_orig_list_idx
        initial_points_3d_optim_list.append(pt3d_obj.point3d.flatten())

        for cam_orig_idx_viewing_pt in active_views:
            kpt_idx_in_cam = pt3d_obj.source_2dpt_idxs[cam_orig_idx_viewing_pt]
            obs_cam_optim_indices.append(cam_orig_to_optim_idx[cam_orig_idx_viewing_pt])
            obs_pt_optim_indices.append(current_optim_pt_idx)
            obs_points_2d_list.append(
                all_keypoints[cam_orig_idx_viewing_pt][kpt_idx_in_cam].pt
            )

        current_optim_pt_idx += 1

    if not initial_points_3d_optim_list or not obs_points_2d_list:
        print(
            "Warning: No 3D points observed by active cameras, or no 2D observations. Skipping BA."
        )
        return points3d_with_views, R_mats, t_vecs

    n_points_optim = len(initial_points_3d_optim_list)
    initial_points_3d_np = np.array(initial_points_3d_optim_list, dtype=np.float32)

    obs_cam_optim_indices_np = np.array(obs_cam_optim_indices, dtype=np.int32)
    obs_pt_optim_indices_np = np.array(obs_pt_optim_indices, dtype=np.int32)
    obs_points_2d_np = np.array(obs_points_2d_list, dtype=np.float32)

    # Combine into a single parameter vector for PyTorch
    initial_params_np = np.hstack(
        (initial_camera_params_np.ravel(), initial_points_3d_np.ravel())
    )

    # 2. PyTorch Setup
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using PyTorch device: {device}")

    params_tensor = torch.tensor(
        initial_params_np, dtype=torch.float32, device=device, requires_grad=True
    )
    cam_indices_tensor = torch.tensor(
        obs_cam_optim_indices_np, dtype=torch.long, device=device
    )
    pt_indices_tensor = torch.tensor(
        obs_pt_optim_indices_np, dtype=torch.long, device=device
    )
    pts_2d_tensor = torch.tensor(obs_points_2d_np, dtype=torch.float32, device=device)
    K_tensor = torch.tensor(
        K_np, dtype=torch.float32, device=device
    )  # K_np should be float

    # Optimizer
    optimizer = optim.Adam([params_tensor], lr=learning_rate)

    print(
        f"Optimizing {n_cameras_optim} cameras, {n_points_optim} points, {len(obs_points_2d_list)} observations."
    )

    # 3. Optimization Loop
    prev_loss = None
    bad_iters = 0

    for i in range(n_iterations):
        optimizer.zero_grad(set_to_none=True)
        reprojection_errors = calculate_reprojection_error_torch(
            params_tensor,
            n_cameras_optim,
            n_points_optim,
            cam_indices_tensor,
            pt_indices_tensor,
            pts_2d_tensor,
            K_tensor,
        )

        if loss_type == "huber":
            abs_err = reprojection_errors.abs()
            quadratic = torch.clamp(abs_err, max=huber_delta)
            linear = abs_err - quadratic
            loss = (0.5 * quadratic.pow(2) + huber_delta * linear).sum()
        else:
            loss = reprojection_errors.pow(2).sum()  # Sum of squared errors

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if (i + 1) % (n_iterations // 10 if n_iterations >= 10 else 1) == 0:
            print(f"Iteration {i+1}/{n_iterations}, Loss: {loss_val:.4e}")

        if early_stop_patience and prev_loss is not None:
            improvement = prev_loss - loss_val
            if improvement < early_stop_tol * max(1.0, prev_loss):
                bad_iters += 1
            else:
                bad_iters = 0

            if bad_iters >= early_stop_patience:
                print(
                    f"Early stopping at iteration {i+1} (no significant improvement)."
                )
                break

        prev_loss = loss_val

    print(f"PyTorch BA finished. Final Loss: {loss.item():.4e}")

    # 4. Retrieve and Update Parameters
    optimized_params_np = params_tensor.detach().cpu().numpy()

    num_cam_params_per_cam = 12
    adj_cam_params_flat = optimized_params_np[
        : n_cameras_optim * num_cam_params_per_cam
    ]
    adj_pts_3d_flat = optimized_params_np[n_cameras_optim * num_cam_params_per_cam :]

    adj_cam_params_optim = adj_cam_params_flat.reshape(
        n_cameras_optim, num_cam_params_per_cam
    )
    adj_pts_3d_optim = adj_pts_3d_flat.reshape(n_points_optim, 3)

    # Update R_mats and t_vecs
    for orig_idx, optim_idx in cam_orig_to_optim_idx.items():
        R_flat = adj_cam_params_optim[optim_idx][:9]
        R = R_flat.reshape(3, 3)

        if apply_orthogonalization:
            # Ensure R is a valid rotation matrix (orthogonalization)
            U, _, Vt = np.linalg.svd(R)
            R_ortho = U @ Vt
            if np.linalg.det(R_ortho) < 0:  # Ensure it's a right-handed system
                Vt[-1, :] *= -1
                R_ortho = U @ Vt
            R_mats[orig_idx] = R_ortho
        else:
            R_mats[orig_idx] = R

        t_vecs[orig_idx] = adj_cam_params_optim[optim_idx][9:].reshape(3, 1)

    # Update points3d_with_views
    for optim_pt_idx, i_orig_list_idx in optim_idx_to_pt3d_orig_list_idx.items():
        points3d_with_views[i_orig_list_idx].point3d = adj_pts_3d_optim[
            optim_pt_idx
        ].reshape(1, 3)

    return points3d_with_views, R_mats, t_vecs
