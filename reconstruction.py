import numpy as np
import random
import cv2
from typing import Tuple, List, Dict, Optional, Union

# Import BoVW functions from utils
from utils import match_images_bovw_tfidf


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


class ReconstructionPipeline:
    def __init__(
        self,
        img_adjacency: np.ndarray,
        matches: Dict[Tuple[int, int], List[cv2.DMatch]],
        keypoints: List[List[cv2.KeyPoint]],
        connected_pairs_,
        K: np.matrix,
        tfidf_histograms: Optional[Dict[int, np.ndarray]] = None,
        use_bovw_for_next_best_pair: bool = False,
    ):
        """
        Initializes the ReconstructionPipeline.

        Args:
            img_adjacency (np.ndarray): Adjacency matrix indicating image connectivity.
            matches (Dict[Tuple[int, int], List[cv2.DMatch]]): Dictionary of filtered matches, where keys are (smaller_idx, larger_idx).
            keypoints (List[List[cv2.KeyPoint]]): List of keypoints for each image.
            connected_pairs_ (List[Tuple[int, int]]): List of connected image pairs.
            K (np.ndarray): Camera intrinsic matrix.
            tfidf_histograms (Optional[Dict[int, np.ndarray]]): Dictionary of TF-IDF histograms for each image.
            use_bovw_for_next_best_pair (bool): Flag to indicate if BoVW should be used for next best pair selection.
        """
        self.img_adjacency = img_adjacency
        self.matches = matches
        self.keypoints = keypoints
        self.connected_pairs = connected_pairs_
        self.K = K
        self.tfidf_histograms = tfidf_histograms
        self.use_bovw_for_next_best_pair = use_bovw_for_next_best_pair

    # --- Image Pair Selection Functions ---
    def best_img_pair(self, top_x_perc: float = 0.2) -> Optional[Tuple[int, int]]:
        # Ensure correct access of matches for num_matches calculation
        # self.matches is a dictionary where keys are (idx1, idx2) with idx1 < idx2
        num_matches = []
        for i, j in self.connected_pairs:
            key = tuple(sorted((i, j)))
            if key in self.matches:
                num_matches.append(len(self.matches[key]))

        if not num_matches:
            return None

        num_matches_sorted = sorted(num_matches, reverse=True)
        min_match_idx = int(len(num_matches_sorted) * top_x_perc)
        min_matches = (
            num_matches_sorted[min_match_idx]
            if min_match_idx < len(num_matches_sorted)
            else 0
        )

        best_rot_angle = 0.0
        best_pair = None

        for i, j in self.connected_pairs:
            key = tuple(sorted((i, j)))
            if key in self.matches and len(self.matches[key]) > min_matches:
                kpts_i, kpts_j, _, _ = self.get_aligned_kpts(
                    i, j, self.keypoints, self.matches
                )
                if len(kpts_i) < 8:  # Need at least 8 points for Essential Matrix
                    continue
                E, _ = cv2.findEssentialMat(
                    kpts_i, kpts_j, self.K, cv2.FM_RANSAC, 0.999, 1.0
                )
                if E is None:
                    continue
                points, R1, t1, mask = cv2.recoverPose(E, kpts_i, kpts_j, self.K)

                if (
                    points == 0 or R1 is None or t1 is None
                ):  # check if pose recovery was successful
                    continue

                rvec, _ = cv2.Rodrigues(R1)
                rot_angle = float(np.sum(np.abs(rvec)))
                if (rot_angle > best_rot_angle or best_pair is None) and points == len(
                    kpts_i
                ):
                    best_rot_angle = rot_angle
                    best_pair = (i, j)

        return best_pair

    def get_aligned_kpts(
        self,
        i: int,
        j: int,
        keypoints: List[List[cv2.KeyPoint]],
        matches: Dict[
            Tuple[int, int], List[cv2.DMatch]
        ],  # Changed type hint to Dict for clarity
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Extract aligned arrays of matched 2D keypoints between two images.
        Handles the case where matches are stored as (smaller_idx, larger_idx) tuples.

        Args:
            i (int): Index of the first image.
            j (int): Index of the second image.
            keypoints (List[List[cv2.KeyPoint]]): List of keypoints per image.
            matches (Dict[Tuple[int, int], List[cv2.DMatch]]): Dictionary where keys are
                                                      (image_idx1, image_idx2) tuples
                                                      with image_idx1 < image_idx2.
            mask (Optional[np.ndarray]): Optional boolean mask to select subset of matches.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[int], List[int]]: pts_i, pts_j arrays of shape (M, 1, 2) of matched keypoint coordinates,
            and idxs_i, idxs_j lists of original keypoint indices in each image.
        """
        # Ensure we access the matches dictionary with the smaller index first
        key = tuple(sorted((i, j)))

        # Check if the key exists in the matches dictionary
        if key not in matches:
            # This indicates no filtered matches found for this pair
            # Or the order was incorrect and not handled
            print(
                f"Warning: No matches found for pair {i, j} (or {j, i}). Skipping get_aligned_kpts."
            )
            return np.array([]).reshape(0, 1, 2), np.array([]).reshape(0, 1, 2), [], []

        dm = matches[key]

        # Determine if (i,j) or (j,i) was the original order of the match (queryIdx, trainIdx)
        # to correctly map queryIdx to image `i` and trainIdx to image `j`.
        reverse_order = False
        if i > j:  # If 'i' was the larger index, it was 'train' in the stored match
            reverse_order = True

        M = len(dm)
        if M == 0:
            return np.array([]).reshape(0, 1, 2), np.array([]).reshape(0, 1, 2), [], []

        if mask is None:
            mask = np.ones(M, dtype=bool)

        pts_i, pts_j = [], []
        idxs_i, idxs_j = [], []

        for k, m in enumerate(dm):
            if not mask[k]:
                continue

            # Adjust queryIdx and trainIdx based on the original order of (i,j) vs sorted(i,j)
            if (
                not reverse_order
            ):  # (i,j) was (smaller, larger) -> matches[i][j] where i is query, j is train
                pt_i = keypoints[i][m.queryIdx].pt
                pt_j = keypoints[j][m.trainIdx].pt
                idxs_i.append(m.queryIdx)
                idxs_j.append(m.trainIdx)
            else:  # (i,j) was (larger, smaller) -> matches[j][i] was stored, so m.queryIdx refers to j, m.trainIdx to i
                pt_i = keypoints[i][
                    m.trainIdx
                ].pt  # Here m.trainIdx corresponds to image i
                pt_j = keypoints[j][
                    m.queryIdx
                ].pt  # Here m.queryIdx corresponds to image j
                idxs_i.append(m.trainIdx)
                idxs_j.append(m.queryIdx)

            pts_i.append(pt_i)
            pts_j.append(pt_j)

        pts_i = np.expand_dims(np.array(pts_i), axis=1)
        pts_j = np.expand_dims(np.array(pts_j), axis=1)

        return pts_i, pts_j, idxs_i, idxs_j

    # --- Triangulation and Reprojection Functions ---

    def triangulate_points_and_reproject(
        self,
        R1: np.ndarray,
        t1: np.ndarray,
        R2: np.ndarray,
        t2: np.ndarray,
        K: np.matrix,
        points3d: List[Point3DWithViews],
        idx1: int,
        idx2: int,
        pts_i: np.ndarray,
        pts_j: np.ndarray,
        idxs_i: List[int],
        idxs_j: List[int],
        compute_reproj: bool = True,
    ) -> Union[
        List[Point3DWithViews],
        Tuple[List[Point3DWithViews], List[Tuple[float, float]], float, float],
    ]:
        """
        Triangulate 3D points from two views and optionally compute reprojection errors.

        Args:
            R1 (np.ndarray): Rotation matrix of image idx1.
            t1 (np.ndarray): Translation vector of image idx1.
            R2 (np.ndarray): Rotation matrix of image idx2.
            t2 (np.ndarray): Translation vector of image idx2.
            K (np.ndarray): Camera intrinsic matrix.
            points3d (List[Point3DWithViews]): List to append new Point3DWithViews.
            idx1 (int): Image index 1.
            idx2 (int): Image index 2.
            pts_i (np.ndarray): Matched keypoint array for image idx1 (2xN).
            pts_j (np.ndarray): Matched keypoint array for image idx2 (2xN).
            idxs_i (List[int]): Corresponding keypoint indices for image idx1.
            idxs_j (List[int]): Corresponding keypoint indices for image idx2.
            compute_reproj (bool): Flag to compute reprojection error.

        Returns:
            Union[List[Point3DWithViews], Tuple[List[Point3DWithViews], List[Tuple[float, float]], float, float]]:
            If compute_reproj is True, returns points3d, errors, avg_error_img1, avg_error_img2.
            Otherwise, returns points3d.
        """
        P_l = K @ np.hstack((R1, t1))
        P_r = K @ np.hstack((R2, t2))

        kpts_i = np.squeeze(pts_i).T.reshape(2, -1)
        kpts_j = np.squeeze(pts_j).T.reshape(2, -1)

        if kpts_i.shape[1] == 0:
            print("Warning: No points to triangulate.")
            if compute_reproj:
                return points3d, [], 0.0, 0.0
            return points3d

        point_4d_hom = cv2.triangulatePoints(P_l, P_r, kpts_i, kpts_j)
        points_3D = cv2.convertPointsFromHomogeneous(point_4d_hom.T)

        for i in range(kpts_i.shape[1]):
            source_2dpt_idxs = {idx1: idxs_i[i], idx2: idxs_j[i]}
            pt = Point3DWithViews(points_3D[i], source_2dpt_idxs)
            points3d.append(pt)

        if compute_reproj:
            kpts_i = kpts_i.T
            kpts_j = kpts_j.T
            rvec_l, _ = cv2.Rodrigues(R1)
            rvec_r, _ = cv2.Rodrigues(R2)

            # Ensure points_3D is float32 and correct shape for projectPoints
            points_3D_float32 = points_3D.astype(np.float32)

            projPoints_l, _ = cv2.projectPoints(
                points_3D_float32, rvec_l, t1, K, distCoeffs=np.array([])
            )
            projPoints_r, _ = cv2.projectPoints(
                points_3D_float32, rvec_r, t2, K, distCoeffs=np.array([])
            )

            delta_l, delta_r = [], []
            for i in range(len(projPoints_l)):
                delta_l.append(abs(projPoints_l[i][0][0] - kpts_i[i][0]))
                delta_l.append(abs(projPoints_l[i][0][1] - kpts_i[i][1]))
                delta_r.append(abs(projPoints_r[i][0][0] - kpts_j[i][0]))
                delta_r.append(abs(projPoints_r[i][0][1] - kpts_j[i][1]))

            avg_error_l = sum(delta_l) / len(delta_l) if delta_l else 0.0
            avg_error_r = sum(delta_r) / len(delta_r) if delta_r else 0.0

            print(
                f"Average reprojection error for just-triangulated points on image {idx1} is:",
                avg_error_l,
                "pixels.",
            )
            print(
                f"Average reprojection error for just-triangulated points on image {idx2} is:",
                avg_error_r,
                "pixels.",
            )

            errors = list(zip(delta_l, delta_r))
            return points3d, errors, avg_error_l, avg_error_r

        return points3d

    def initialize_reconstruction(
        self, img_idx1: int, img_idx2: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Point3DWithViews]]:
        kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs = self.get_aligned_kpts(
            img_idx1, img_idx2, self.keypoints, self.matches
        )

        if len(kpts_i) < 8:  # Need at least 8 points for Fundamental Matrix
            print(
                f"Warning: Not enough aligned keypoints ({len(kpts_i)}) for initial reconstruction between {img_idx1} and {img_idx2}."
            )
            # You might want to raise an error or handle this more robustly in main.py
            return None, None, None, None, []  # Return empty or error indicators

        E, mask = cv2.findEssentialMat(
            kpts_i, kpts_j, self.K, cv2.FM_RANSAC, 0.999, 1.0
        )
        if E is None:
            print(
                f"Error: Essential Matrix estimation failed for pair {img_idx1}, {img_idx2}."
            )
            return None, None, None, None, []

        points, R1, t1, pose_mask = cv2.recoverPose(
            E, kpts_i, kpts_j, self.K, mask=mask
        )

        # Check if recoverPose was successful and a valid pose was recovered
        if points == 0 or R1 is None or t1 is None:
            print(f"Error: recoverPose failed for pair {img_idx1}, {img_idx2}.")
            return None, None, None, None, []

        assert abs(np.linalg.det(R1)) - 1 < 1e-7

        R0 = np.eye(3)
        t0 = np.zeros((3, 1))

        points3d_with_views = []
        # Filter kpts_i, kpts_j, idxs_i, idxs_j by pose_mask to triangulate only inliers
        kpts_i_inliers = kpts_i[pose_mask.ravel() == 1]
        kpts_j_inliers = kpts_j[pose_mask.ravel() == 1]
        idxs_i_inliers = [
            kpts_i_idxs[i] for i in range(len(kpts_i_idxs)) if pose_mask[i] == 1
        ]
        idxs_j_inliers = [
            kpts_j_idxs[i] for i in range(len(kpts_j_idxs)) if pose_mask[i] == 1
        ]

        points3d_with_views = self.triangulate_points_and_reproject(
            R0,
            t0,
            R1,
            t1,
            self.K,
            points3d_with_views,
            img_idx1,
            img_idx2,
            kpts_i_inliers,
            kpts_j_inliers,
            idxs_i_inliers,
            idxs_j_inliers,
            compute_reproj=False,
        )

        return R0, t0, R1, t1, points3d_with_views

    def get_idxs_in_correct_order(self, idx1, idx2):
        """First idx must be smaller than second when using upper-triangular arrays (matches, keypoints)"""
        if idx1 < idx2:
            return idx1, idx2
        else:
            return idx2, idx1

    def images_adjacent(self, i, j, img_adjacency):
        """Return true if both images view the same scene (have enough matches)."""
        if img_adjacency[i, j] == 1 or img_adjacency[j, i] == 1:
            return True
        else:
            return False

    def check_and_get_unresected_point(
        self, resected_kpt_idx, match, resected_idx, unresected_idx
    ):
        """
        Check if a 3D point seen by the given resected image is involved in a match to the unresected image
        and is therefore usable for Pnp.

        :param resected_kpt_idx: Index of keypoint in keypoints list for resected image
        :param match: cv2.DMatch object
        :resected_idx: Index of the resected image
        :unresected_idx: Index of the unresected image
        """
        # Determine which index in the match object (queryIdx or trainIdx) corresponds to resected_idx
        # and which to unresected_idx. This depends on how the match was originally stored.
        # Assuming matches[min(idx1, idx2), max(idx1, idx2)] was created such that
        # queryIdx belongs to the first index in the tuple and trainIdx to the second.

        # Find the correct match object to check
        if (
            min(resected_idx, unresected_idx) == resected_idx
        ):  # Stored as (resected, unresected)
            if resected_kpt_idx == match.queryIdx:
                unresected_kpt_idx = match.trainIdx
                success = True
                return unresected_kpt_idx, success
        else:  # Stored as (unresected, resected)
            if resected_kpt_idx == match.trainIdx:
                unresected_kpt_idx = match.queryIdx
                success = True
                return unresected_kpt_idx, success
        return None, False

    def get_correspondences_for_pnp(
        self, resected_idx, unresected_idx, pts3d, matches, keypoints
    ):
        """
        Returns index aligned lists of 3D and 2D points to be used for Pnp. For each 3D point check if it is seen
        by the resected image, if so check if there is a match for it between the resected and unresected image.
        If so that point will be used in Pnp. Also keeps track of matches that do not have associated 3D points,
        and therefore need to be triangulated.

        :param resected_idx: Index of resected image to be used in Pnp
        :param unresected_idx Index of unresected image to be used in Pnp
        :param pts3d: List of Point3D_with_views objects
        :param matches: Dictionary of matches where matches[(i,j)] is the list of cv2.DMatch objects for images i and j (i<j)
        :param keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
        """

        match_key = tuple(sorted((resected_idx, unresected_idx)))
        if match_key not in matches:
            print(
                f"Warning: No filtered matches found for pair {resected_idx, unresected_idx} during PnP correspondence gathering. Returning empty lists."
            )
            return pts3d, [], [], np.array([])  # Return empty lists if no matches

        current_matches_list = matches[match_key]
        triangulation_status = np.ones(
            len(current_matches_list), dtype=bool
        )  # if triangulation_status[x] = True, then matches[x] used for triangulation

        pts3d_for_pnp = []
        pts2d_for_pnp = []

        for pt3d in pts3d:
            # Check if this 3D point is seen by the resected image
            if resected_idx not in pt3d.source_2dpt_idxs:
                continue

            resected_kpt_idx_in_3d_point = pt3d.source_2dpt_idxs[resected_idx]

            found_match_for_pnp = False
            for k, match in enumerate(current_matches_list):
                # Determine which keypoint index in the match object belongs to resected_idx
                # and which belongs to unresected_idx, based on how the match was stored.

                # If match_key was (resected_idx, unresected_idx)
                if (
                    match_key[0] == resected_idx
                ):  # resected_idx is the query image in this match
                    if match.queryIdx == resected_kpt_idx_in_3d_point:
                        unresected_kpt_idx_in_match = match.trainIdx
                        # Check if this match refers to the resected keypoint from the 3D point
                        # and that the corresponding train keypoint in match is what we need for unresected

                        # Add new 2d/3d correspondences to 3D point object
                        # Only add if it's not already there
                        if unresected_idx not in pt3d.source_2dpt_idxs:
                            pt3d.source_2dpt_idxs[unresected_idx] = (
                                unresected_kpt_idx_in_match
                            )

                        pts3d_for_pnp.append(pt3d.point3d)
                        pts2d_for_pnp.append(
                            keypoints[unresected_idx][unresected_kpt_idx_in_match].pt
                        )
                        triangulation_status[k] = (
                            False  # This match is used for PnP, not triangulation
                        )
                        found_match_for_pnp = True
                        break  # Move to next 3D point
                # If match_key was (unresected_idx, resected_idx)
                elif (
                    match_key[1] == resected_idx
                ):  # resected_idx is the train image in this match
                    if match.trainIdx == resected_kpt_idx_in_3d_point:
                        unresected_kpt_idx_in_match = match.queryIdx

                        if unresected_idx not in pt3d.source_2dpt_idxs:
                            pt3d.source_2dpt_idxs[unresected_idx] = (
                                unresected_kpt_idx_in_match
                            )

                        pts3d_for_pnp.append(pt3d.point3d)
                        pts2d_for_pnp.append(
                            keypoints[unresected_idx][unresected_kpt_idx_in_match].pt
                        )
                        triangulation_status[k] = (
                            False  # This match is used for PnP, not triangulation
                        )
                        found_match_for_pnp = True
                        break  # Move to next 3D point

        return pts3d, pts3d_for_pnp, pts2d_for_pnp, triangulation_status

    def do_pnp(
        self,
        pts3d_for_pnp: List[np.ndarray],
        pts2d_for_pnp: List[Tuple[float, float]],
        K: np.matrix,
        iterations: int = 200,
        reprojThresh: float = 5,
    ):
        """
        Performs PnP using cv2.solvePnPRansac to estimate the pose of a new camera.

        Args:
            pts3d_for_pnp: List of 3D point coordinates (each element typically np.ndarray shape (3,) or (1,3)).
            pts2d_for_pnp: List of corresponding 2D image point coordinates (each element tuple (x,y)).
            K: Camera intrinsic matrix.
            iterations: Number of RANSAC iterations for solvePnPRansac.
            reprojThresh: RANSAC inlier threshold (pixel distance). solvePnPRansac uses squared error.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Estimated rotation matrix (3x3) and translation vector (3x1),
                                                              or (None, None) if PnP fails.
        """
        if not pts3d_for_pnp or not pts2d_for_pnp:
            print("Error: Empty 3D or 2D points passed to do_pnp.")
            return None, None

        if len(pts3d_for_pnp) != len(pts2d_for_pnp):
            print("Error: Mismatch in the number of 3D and 2D points for PnP.")
            return None, None

        min_pnp_points = 6  # Changed from 4 to 6 as per main.py constant
        if len(pts3d_for_pnp) < min_pnp_points:
            print(
                f"Warning: Only {len(pts3d_for_pnp)} points provided for PnP. Needs at least {min_pnp_points}. PnP may fail."
            )
            return None, None  # PnP likely to fail with too few points

        try:
            # Ensure input arrays are of correct shape and type
            # pts3d_for_pnp contains elements that are (1,3) arrays, reshape to (3,) for objectPoints
            object_points = np.array(
                [p.ravel() for p in pts3d_for_pnp], dtype=np.float32
            )
            image_points = np.array(pts2d_for_pnp, dtype=np.float32).reshape(
                -1, 1, 2
            )  # Ensure 1x2 per point
        except Exception as e:
            print(f"Error formatting points for PnP: {e}")
            return None, None

        if object_points.shape[0] < min_pnp_points:
            print(
                f"Error: Not enough points ({object_points.shape[0]}) for PnP after formatting."
            )
            return None, None

        try:
            # cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec=None, tvec=None, useExtrinsicGuess=None, flags=None, rvec=None, tvec=None)
            # The signature can be tricky. Use flags for method selection.
            # reprojThresh is passed as `reprojError` argument for SOLVEPNP_RANSAC

            # Using SOLVEPNP_P3P or SOLVEPNP_EPNP as more robust alternatives if SOLVEPNP_ITERATIVE fails,
            # but SOLVEPNP_RANSAC is common. Sticking to solvePnPRansac for consistency with your code.

            # Ensure distCoeffs is empty array if not used
            distCoeffs = np.array([])

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                K,
                distCoeffs,
                reprojectionError=reprojThresh,  # This is the correct parameter for RANSAC threshold
                iterationsCount=iterations,  # Max iterations for RANSAC
                flags=cv2.SOLVEPNP_P3P,  # Using P3P within RANSAC, or SOLVEPNP_EPNP
            )
        except Exception as e:
            print(f"Error during cv2.solvePnPRansac call: {e}")
            print(
                f"Debug Info - Object Points Shape: {object_points.shape}, Image Points Shape: {image_points.shape}"
            )
            print(
                f"Debug Info - K Matrix Shape: {K.shape}, Reproj Threshold: {reprojThresh}, Iterations: {iterations}"
            )
            return None, None

        if success and rvec is not None and tvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            print(
                f"solvePnPRansac successful. Inliers: {len(inliers) if inliers is not None else 'N/A'}/{len(object_points)}"
            )
            return R, tvec
        else:
            print("solvePnPRansac failed.")
            if not success:
                print("  - Success flag was False.")
            if rvec is None:
                print("  - rvec was None.")
            if tvec is None:
                print("  - tvec was None.")
            return None, None

    def prep_for_reproj(self, img_idx, points3d_with_views, keypoints):
        """
        Returns aligned vectors of 2D and 3D points to be used for reprojection

        :param img_idx: Index of image for which reprojection errors are desired
        :param points3d_with_views: List of Point3D_with_views objects. Will have new points appended to it
        :param keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
        """
        points_3d = []
        points_2d = []
        pt3d_idxs = (
            []
        )  # Not directly used in this function's return, but good for debugging/tracking
        i = 0
        for pt3d in points3d_with_views:
            if img_idx in pt3d.source_2dpt_idxs.keys():
                pt3d_idxs.append(i)
                # Ensure 3D point is correctly shaped for projectPoints, (N,1,3) or (N,3)
                points_3d.append(pt3d.point3d.flatten())  # Flatten to (3,)
                kpt_idx = pt3d.source_2dpt_idxs[img_idx]
                points_2d.append(keypoints[img_idx][kpt_idx].pt)
            i += 1

        # Convert lists to NumPy arrays and ensure correct shapes
        # points_3d should be (N, 3) for cv2.projectPoints
        # points_2d should be (N, 2)
        return (
            np.array(points_3d, dtype=np.float32),
            np.array(points_2d, dtype=np.float32),
            pt3d_idxs,
        )

    def calculate_reproj_errors(self, projPoints, points_2d):
        """
        Calculate reprojection errors (L1) between projected points and ground truth (keypoint coordinates)

        :param projPoints: list of index aligned  projected points (N, 1, 2)
        :param points_2d: list of index aligned corresponding keypoint coordinates (N, 2)
        """
        # Ensure projPoints are squeezed to (N, 2) for direct comparison
        projPoints_squeezed = np.squeeze(projPoints)

        if (
            len(projPoints_squeezed.shape) == 1
        ):  # Handle case where only one point was projected
            projPoints_squeezed = projPoints_squeezed.reshape(1, 2)

        assert (
            projPoints_squeezed.shape == points_2d.shape
        ), f"Shape mismatch: projPoints {projPoints_squeezed.shape}, points_2d {points_2d.shape}"

        delta = np.abs(
            projPoints_squeezed - points_2d
        )  # Element-wise absolute difference

        # Calculate sum of absolute errors for x and y components for each point, then overall average
        average_delta = np.mean(delta)

        return average_delta, delta.tolist()  # Return errors as a list of lists/tuples

    def get_reproj_errors(
        self, img_idx, points3d_with_views, R, t, K, keypoints, distCoeffs=np.array([])
    ):
        """
        Project all 3D points seen in image[img_idx] onto it, return reprojection errors and average error

        :param img_idx: Index of image for which reprojection errors are desired
        :param points3d_with_views: List of Point3D_with_views objects. Will have new points appended to it
        :param R: Rotation matrix
        :param t: Translation vector
        :param K: Intrinsics matrix
        :param keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
        """
        points_3d, points_2d, pt3d_idxs = self.prep_for_reproj(
            img_idx, points3d_with_views, keypoints
        )

        if points_3d.shape[0] == 0:
            return np.array([]), np.array([]), 0.0, []  # No points to reproject

        rvec, _ = cv2.Rodrigues(R)
        projPoints, _ = cv2.projectPoints(points_3d, rvec, t, K, distCoeffs=distCoeffs)

        avg_error, errors = self.calculate_reproj_errors(projPoints, points_2d)

        return points_3d, points_2d, avg_error, errors

    def test_reproj_pnp_points(
        self, pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K, rep_thresh=5
    ):
        """
        Reprojects points fed into Pnp back onto camera whose R and t were just obtained via Pnp.
        Used to assess how good the resection was.

        :param pts3d_for_pnp: List of axis aligned 3D points (each element (1,3) or (3,))
        :param pts2d_for_pnp: List of axis aligned 2D points (each element (x,y))
        :param R_new: Rotation matrix of newly resected image
        :param t_new: Translation vector of newly resected image
        :param rep_thresh: Number of pixels reprojected points must be within to qualify as inliers
        """
        if not pts3d_for_pnp or not pts2d_for_pnp:
            return [], [], 0.0, 0.0  # Return empty if no points

        # Ensure consistent input format for projectPoints
        object_points = np.array([p.ravel() for p in pts3d_for_pnp], dtype=np.float32)
        image_points = np.array(pts2d_for_pnp, dtype=np.float32)  # Shape (N, 2)

        rvec_new, _ = cv2.Rodrigues(R_new)
        projpts_raw, _ = cv2.projectPoints(
            object_points, rvec_new, t_new, K, distCoeffs=np.array([])
        )

        projpts = np.squeeze(projpts_raw)  # Shape (N, 2)

        if projpts.ndim == 1 and projpts.size > 0:  # Handle single point case
            projpts = projpts.reshape(1, 2)

        errors = np.abs(projpts - image_points)  # Element-wise absolute differences

        # Calculate sum of absolute errors for x and y components for each point
        sum_errors_per_point = np.sum(errors, axis=1)  # Shape (N,)

        # Identify inliers based on combined x+y error
        inliers_mask = sum_errors_per_point < (
            2 * rep_thresh
        )  # Each component < rep_thresh, so sum < 2*rep_thresh

        avg_err = (
            np.mean(sum_errors_per_point) / 2.0
            if len(sum_errors_per_point) > 0
            else 0.0
        )  # Average per component
        perc_inliers = (
            np.sum(inliers_mask) / len(pts3d_for_pnp) if len(pts3d_for_pnp) > 0 else 0.0
        )

        return errors.tolist(), projpts.tolist(), avg_err, perc_inliers

    def get_pair_score(
        self,
        unresected_idx: int,
        resected_idx: int,
        points3d_with_views: List[Point3DWithViews],
    ) -> float:
        """
        Calculates a score for pairing an unresected image with a resected one,
        optionally incorporating BoVW TF-IDF similarity.
        Higher score is better.

        Args:
            unresected_idx (int): Index of the unresected image.
            resected_idx (int): Index of the resected image.
            points3d_with_views (List[Point3DWithViews]): Current list of reconstructed 3D points.

        Returns:
            float: The calculated score for the pair. Returns -1 if the pair is invalid (e.g., no matches).
        """
        score = 0.0

        # Ensure indices are ordered correctly for accessing self.matches
        match_key = tuple(sorted((unresected_idx, resected_idx)))

        if match_key not in self.matches or len(self.matches[match_key]) == 0:
            return -1.0  # No direct matches, invalid pair

        # Factor 1: Number of direct feature matches between this pair
        num_direct_matches = len(self.matches[match_key])
        score += num_direct_matches * 1.0  # Weight for direct matches (can be adjusted)

        # Factor 2: Number of existing 3D points visible in the resected image
        # that are also visible and matched in the unresected image.
        common_3d_points_count = 0
        for pt3d in points3d_with_views:
            if (
                resected_idx in pt3d.source_2dpt_idxs
            ):  # If the 3D point is visible in the resected image
                resected_kpt_idx_in_3d_point = pt3d.source_2dpt_idxs[resected_idx]

                for match in self.matches[match_key]:
                    # Determine which keypoint index in the match object belongs to resected_idx
                    # and which belongs to unresected_idx, based on how the match was stored.
                    if (
                        match_key[0] == resected_idx
                    ):  # resected_idx is query in this match
                        if match.queryIdx == resected_kpt_idx_in_3d_point:
                            common_3d_points_count += 1
                            break  # Found for this 3D point, move to next 3D point
                    elif (
                        match_key[1] == resected_idx
                    ):  # resected_idx is train in this match
                        if match.trainIdx == resected_kpt_idx_in_3d_point:
                            common_3d_points_count += 1
                            break  # Found for this 3D point, move to next 3D point

        score += (
            common_3d_points_count * 5.0
        )  # Give more weight to existing 3D points (can be adjusted)

        # Factor 3: BoVW TF-IDF Similarity (if enabled and data available)
        if self.use_bovw_for_next_best_pair and self.tfidf_histograms:
            hist_unres = self.tfidf_histograms.get(unresected_idx)
            hist_res = self.tfidf_histograms.get(resected_idx)

            if (
                hist_unres is not None
                and hist_res is not None
                and hist_unres.size > 0
                and hist_res.size > 0
            ):
                bovw_sim = match_images_bovw_tfidf(
                    hist_unres, hist_res, metric="cosine"
                )
                # Add BoVW similarity to score. Scale it to be meaningful alongside other factors.
                # A cosine similarity of 1.0 would add, e.g., 100 to the score.
                score += bovw_sim * 100.0  # Adjust weight as needed
            # else:
            # print(f"Debug: Missing BoVW histograms for pair ({unresected_idx}, {resected_idx}).")

        return score

    def next_img_pair_to_grow_reconstruction_scored(
        self,
        n_imgs,
        resected_imgs,
        unresected_imgs,
        img_adjacency,
        matches,
        keypoints,
        points3d_with_views,
    ):
        """
        Selects the next unresected image and its best resected partner based on a scoring system.
        Incorporates BoVW TF-IDF similarity if enabled.
        """
        if not unresected_imgs:
            return None, None, False

        best_unresected_idx = -1
        best_resected_partner_idx = -1
        highest_score = -1.0  # Initialize with a low float value

        # Iterate over all unresected images
        for unres_idx in unresected_imgs:
            # For each unresected image, find its best partner among the resected images
            current_best_partner_for_unres = -1
            current_highest_partner_score = -1.0

            for res_idx in resected_imgs:
                if unres_idx == res_idx:
                    continue

                # Check if direct matches exist using the FeatureMatcher's stored filtered_matches structure
                # The key for matches is always (smaller_idx, larger_idx)
                u_idx_check, r_idx_check = min(unres_idx, res_idx), max(
                    unres_idx, res_idx
                )
                if (u_idx_check, r_idx_check) not in matches or not matches[
                    (u_idx_check, r_idx_check)
                ]:
                    continue  # No direct matches, skip this resected partner

                # Calculate score for this pair using the updated get_pair_score method
                score = self.get_pair_score(unres_idx, res_idx, points3d_with_views)

                if score > current_highest_partner_score:
                    current_highest_partner_score = score
                    current_best_partner_for_unres = res_idx

            # If this unresected image found a good partner and its score is the best so far
            if (
                current_best_partner_for_unres != -1
                and current_highest_partner_score > highest_score
            ):
                highest_score = current_highest_partner_score
                best_unresected_idx = unres_idx
                best_resected_partner_idx = current_best_partner_for_unres

        chosen_prepend = (
            False  # Default, can be adjusted by more complex logic if needed
        )
        if best_unresected_idx != -1:
            # Basic prepend logic (can be made more sophisticated)
            # For example, if best_unresected_idx is numerically smaller than the first resected image, prepend.
            if resected_imgs and best_unresected_idx < resected_imgs[0]:
                chosen_prepend = True
            else:
                chosen_prepend = False
            print(
                f"Scored selection: Unresected {best_unresected_idx} with Resected {best_resected_partner_idx} (Score: {highest_score:.2f})"
            )
            return best_resected_partner_idx, best_unresected_idx, chosen_prepend
        else:
            print(
                "Warning: No suitable image pair found by next_img_pair_to_grow_reconstruction_scored."
            )
            if (
                unresected_imgs
            ):  # If still images left, try to pick one to allow main.py to remove it if it fails
                # Fallback: pick the first available unresected and a random resected one
                # This is a very basic fallback.
                unres_fallback = unresected_imgs[0]
                res_fallback = random.choice(resected_imgs) if resected_imgs else None
                if res_fallback is not None:
                    print(
                        f"Fallback selection: Unresected {unres_fallback} with Resected {res_fallback}"
                    )
                    return res_fallback, unres_fallback, False
            return None, None, False  # Signal failure to find a pair
