import numpy as np
import random
import cv2
from typing import Tuple, List, Dict, Optional, Union

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
    def __init__(self, img_adjacency: np.ndarray, matches: List[List[List[cv2.DMatch]]], keypoints: List[List[cv2.KeyPoint]], connected_pairs_, K:np.matrix):
        """_summary_

        Args:
            img_adjacency (np.ndarray): _description_
            matches (List[List[List[cv2.DMatch]]]): _description_
            keypoints (List[List[cv2.KeyPoint]]): _description_
            connected_pairs_ ()
            K (np.ndarray): _description_
        """
        self.img_adjacency = img_adjacency
        self.matches = matches
        self.keypoints = keypoints
        self.connected_pairs = connected_pairs_
        self.K = K
        
    # --- Image Pair Selection Functions ---
    def best_img_pair(self,
                    top_x_perc: float = 0.2) -> Optional[Tuple[int, int]]:
        num_matches = [len(self.matches[i, j]) for i, j in self.connected_pairs if self.img_adjacency[i, j] == 1]

        if not num_matches:
            return None

        num_matches_sorted = sorted(num_matches, reverse=True)
        min_match_idx = int(len(num_matches_sorted) * top_x_perc)
        min_matches = num_matches_sorted[min_match_idx]

        best_rot_angle = 0.0
        best_pair = None

        for i, j in self.connected_pairs:
            if self.img_adjacency[i, j] == 1 and len(self.matches[i, j]) > min_matches:
                kpts_i, kpts_j, _, _ = self.get_aligned_kpts(i, j, self.keypoints, self.matches)
                E, _ = cv2.findEssentialMat(kpts_i, kpts_j, self.K, cv2.FM_RANSAC, 0.999, 1.0)
                points, R1, t1, mask = cv2.recoverPose(E, kpts_i, kpts_j, self.K)
                rvec, _ = cv2.Rodrigues(R1)
                rot_angle = float(np.sum(np.abs(rvec)))
                if (rot_angle > best_rot_angle or best_pair is None) and points == len(kpts_i):
                    best_rot_angle = rot_angle
                    best_pair = (i, j)

        return best_pair

    def get_aligned_kpts(self, 
                         i: int,
                        j: int,
                        keypoints: List[List[cv2.KeyPoint]],
                        matches: List[List[List[cv2.DMatch]]],
                        mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Extract aligned arrays of matched 2D keypoints between two images.

        Args:
            i (int): Index of the first image.
            j (int): Index of the second image.
            keypoints (List[List[cv2.KeyPoint]]): List of keypoints per image.
            matches (List[List[List[cv2.DMatch]]]): Match matrix between images.
            mask (Optional[np.ndarray]): Optional boolean mask to select subset of matches.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[int], List[int]]: pts_i, pts_j arrays of shape (M, 1, 2) of matched keypoint coordinates,
            and idxs_i, idxs_j lists of original keypoint indices in each image.
        """
        dm = matches[i, j]
        M = len(dm)
        if mask is None:
            mask = np.ones(M, dtype=bool)

        pts_i, pts_j = [], []
        idxs_i, idxs_j = [], []

        for k, m in enumerate(dm):
            if not mask[k]:
                continue
            pt_i = keypoints[i][m.queryIdx].pt
            pt_j = keypoints[j][m.trainIdx].pt
            pts_i.append(pt_i)
            pts_j.append(pt_j)
            idxs_i.append(m.queryIdx)
            idxs_j.append(m.trainIdx)

        pts_i = np.expand_dims(np.array(pts_i), axis=1)
        pts_j = np.expand_dims(np.array(pts_j), axis=1)

        return pts_i, pts_j, idxs_i, idxs_j

    # --- Triangulation and Reprojection Functions ---

    def triangulate_points_and_reproject(self, 
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
                                        compute_reproj: bool = True) -> Union[List[Point3DWithViews], Tuple[List[Point3DWithViews], List[Tuple[float, float]], float, float]]:
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
            projPoints_l, _ = cv2.projectPoints(points_3D, rvec_l, t1, K, distCoeffs=np.array([]))
            projPoints_r, _ = cv2.projectPoints(points_3D, rvec_r, t2, K, distCoeffs=np.array([]))

            delta_l, delta_r = [], []
            for i in range(len(projPoints_l)):
                delta_l.append(abs(projPoints_l[i][0][0] - kpts_i[i][0]))
                delta_l.append(abs(projPoints_l[i][0][1] - kpts_i[i][1]))
                delta_r.append(abs(projPoints_r[i][0][0] - kpts_j[i][0]))
                delta_r.append(abs(projPoints_r[i][0][1] - kpts_j[i][1]))

            avg_error_l = sum(delta_l) / len(delta_l)
            avg_error_r = sum(delta_r) / len(delta_r)

            print(f"Average reprojection error for just-triangulated points on image {idx1} is:", avg_error_l, "pixels.")
            print(f"Average reprojection error for just-triangulated points on image {idx2} is:", avg_error_r, "pixels.")

            errors = list(zip(delta_l, delta_r))
            return points3d, errors, avg_error_l, avg_error_r

        return points3d

    def initialize_reconstruction(self,
                                img_idx1: int,
                                img_idx2: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Point3DWithViews]]:
        kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs = self.get_aligned_kpts(img_idx1, img_idx2, self.keypoints, self.matches)
        E, _ = cv2.findEssentialMat(kpts_i, kpts_j, self.K, cv2.FM_RANSAC, 0.999, 1.0)
        points, R1, t1, mask = cv2.recoverPose(E, kpts_i, kpts_j, self.K)
        assert abs(np.linalg.det(R1)) - 1 < 1e-7

        R0 = np.eye(3)
        t0 = np.zeros((3, 1))

        points3d_with_views = []
        points3d_with_views = self.triangulate_points_and_reproject(
            R0, t0, R1, t1, self.K, points3d_with_views, img_idx1, img_idx2, kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs, compute_reproj=False)

        return R0, t0, R1, t1, points3d_with_views

    def get_idxs_in_correct_order(self, idx1, idx2):
        """First idx must be smaller than second when using upper-triangular arrays (matches, keypoints)"""
        if idx1 < idx2: return idx1, idx2
        else: return idx2, idx1

    def images_adjacent(self, i, j, img_adjacency):
        """Return true if both images view the same scene (have enough matches)."""
        if img_adjacency[i,j] == 1 or img_adjacency[j,i] == 1:
            return True
        else:
            return False

    def check_and_get_unresected_point(self, resected_kpt_idx, match, resected_idx, unresected_idx):
        """
        Check if a 3D point seen by the given resected image is involved in a match to the unresected image
        and is therefore usable for Pnp.

        :param resected_kpt_idx: Index of keypoint in keypoints list for resected image
        :param match: cv2.Dmatch object
        :resected_idx: Index of the resected image
        :unresected_idx: Index of the unresected image
        """
        if resected_idx < unresected_idx:
            if resected_kpt_idx == match.queryIdx:
                unresected_kpt_idx = match.trainIdx
                success = True
                return unresected_kpt_idx, success
            else:
                return None, False
        elif unresected_idx < resected_idx:
            if resected_kpt_idx == match.trainIdx:
                unresected_kpt_idx = match.queryIdx
                success = True
                return unresected_kpt_idx, success
            else:
                return None, False

    def get_correspondences_for_pnp(self, resected_idx, unresected_idx, pts3d, matches, keypoints):
        """
        Returns index aligned lists of 3D and 2D points to be used for Pnp. For each 3D point check if it is seen
        by the resected image, if so check if there is a match for it between the resected and unresected image.
        If so that point will be used in Pnp. Also keeps track of matches that do not have associated 3D points,
        and therefore need to be triangulated.

        :param resected_idx: Index of resected image to be used in Pnp
        :param unresected_idx Index of unresected image to be used in Pnp
        :param pts3d: List of Point3D_with_views objects
        :param matches: List of lists of lists where matches[i][j][k] is the kth cv2.Dmatch object for images i and j
        :param keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
        """
        idx1, idx2 = self.get_idxs_in_correct_order(resected_idx, unresected_idx)
        triangulation_status = np.ones(len(matches[idx1, idx2])) # if triangulation_status[x] = 1, then matches[x] used for triangulation
        pts3d_for_pnp = []
        pts2d_for_pnp = []
        for pt3d in pts3d:
            if resected_idx not in pt3d.source_2dpt_idxs: continue
            resected_kpt_idx = pt3d.source_2dpt_idxs[resected_idx]
            for k in range(len(matches[idx1, idx2])):
                unresected_kpt_idx, success = self.check_and_get_unresected_point(resected_kpt_idx, matches[idx1, idx2][k], resected_idx, unresected_idx)
                if not success: continue
                pt3d.source_2dpt_idxs[unresected_idx] = unresected_kpt_idx #Add new 2d/3d correspondences to 3D point object
                pts3d_for_pnp.append(pt3d.point3d)
                pts2d_for_pnp.append(keypoints[unresected_idx][unresected_kpt_idx].pt)
                triangulation_status[k] = 0

        return pts3d, pts3d_for_pnp, pts2d_for_pnp, triangulation_status

    def do_pnp(self, pts3d_for_pnp: List[np.ndarray], pts2d_for_pnp: List[Tuple[float, float]], K: np.matrix, iterations: int = 200, reprojThresh: float = 5):
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

        min_pnp_points = 4
        if len(pts3d_for_pnp) < min_pnp_points:
            print(f"Warning: Only {len(pts3d_for_pnp)} points provided for PnP. Needs at least {min_pnp_points}. PnP may fail.")

        try:
            object_points = np.array([p.reshape(3) for p in pts3d_for_pnp], dtype=np.float32)
            image_points = np.array(pts2d_for_pnp, dtype=np.float32).reshape(-1, 2)
        except Exception as e:
            print(f"Error formatting points for PnP: {e}")
            return None, None

        if object_points.shape[0] < min_pnp_points:
             print(f"Error: Not enough points ({object_points.shape[0]}) for PnP after formatting.")
             return None, None

        
        try:
            # Pass all arguments positionally.
            # rvec, tvec, and inliers are outputs; pass None for initial placeholders.
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,             # 1: objectPoints
                image_points,              # 2: imagePoints
                K,                         # 3: cameraMatrix
                cv2.SOLVEPNP_ITERATIVE                      # 6: tvec (Output: initial guess for T vec, or None)
            )
        except Exception as e:
            print(f"Error during cv2.solvePnPRansac call: {e}")
            print(f"Debug Info - Object Points Shape: {object_points.shape}, Image Points Shape: {image_points.shape}")
            print(f"Debug Info - K Matrix Shape: {K.shape}, Iterations set in UsacParams: {iterations}, Reproj Threshold set in UsacParams: {reprojThresh}")
            print("Ensure cv2.UsacParams is correctly initialized and passed. Also, check if your OpenCV build supports the USAC methods specified.")
            return None, None


        if success and rvec is not None and tvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            print(f"solvePnPRansac successful. Inliers: {len(inliers) if inliers is not None else 'N/A'}/{len(object_points)}")
            return R, tvec
        else:
            print("solvePnPRansac failed.")
            if not success: print("  - Success flag was False.")
            if rvec is None: print("  - rvec was None.")
            if tvec is None: print("  - tvec was None.")
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
        pt3d_idxs = []
        i = 0
        for pt3d in points3d_with_views:
            if img_idx in pt3d.source_2dpt_idxs.keys():
                pt3d_idxs.append(i)
                points_3d.append(pt3d.point3d)
                kpt_idx = pt3d.source_2dpt_idxs[img_idx]
                points_2d.append(keypoints[img_idx][kpt_idx].pt)
            i += 1

        return np.array(points_3d), np.array(points_2d), pt3d_idxs

    def calculate_reproj_errors(self, projPoints, points_2d):
        """
        Calculate reprojection errors (L1) between projected points and ground truth (keypoint coordinates)

        :param projPoints: list of index aligned  projected points
        :param points_2d: list of index aligned corresponding keypoint coordinates
        """
        assert len(projPoints) == len(points_2d)
        delta = []
        for i in range(len(projPoints)):
            delta.append(abs(projPoints[i] - points_2d[i]))

        average_delta = sum(delta)/len(delta) # 2-vector, average error for x and y coord
        average_delta = (average_delta[0] + average_delta[1])/2 # average error overall

        return average_delta, delta

    def get_reproj_errors(self, img_idx, points3d_with_views, R, t, K, keypoints, distCoeffs=np.array([])):
        """
        Project all 3D points seen in image[img_idx] onto it, return reprojection errors and average error

        :param img_idx: Index of image for which reprojection errors are desired
        :param points3d_with_views: List of Point3D_with_views objects. Will have new points appended to it
        :param R: Rotation matrix
        :param t: Translation vector
        :param K: Intrinsics matrix
        :param keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
        """
        points_3d, points_2d, pt3d_idxs = self.prep_for_reproj(img_idx, points3d_with_views, keypoints)
        rvec, _ = cv2.Rodrigues(R)
        projPoints, _ = cv2.projectPoints(points_3d, rvec, t, K, distCoeffs=distCoeffs)
        projPoints = np.squeeze(projPoints)
        avg_error, errors = self.calculate_reproj_errors(projPoints, points_2d)

        return points_3d, points_2d, avg_error, errors

    def test_reproj_pnp_points(self, pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K, rep_thresh=5):
        """
        Reprojects points fed into Pnp back onto camera whose R and t were just obtained via Pnp.
        Used to assess how good the resection was.

        :param pts3d_for_pnp: List of axis aligned 3D points
        :param pts2d_for_pnp: List of axis aligned 2D points
        :param R_new: Rotation matrix of newly resected image
        :param t_new: Translation vector of newly resected image
        :param rep_thresh: Number of pixels reprojected points must be within to qualify as inliers
        """
        errors = []
        projpts = []
        inliers = []
        for i in range(len(pts3d_for_pnp)):
            Xw = pts3d_for_pnp[i][0]
            Xr = np.dot(R_new, Xw).reshape(3,1)
            Xc = Xr + t_new
            x = np.dot(K, Xc)
            x /= x[2]
            errors.append([np.float64(x[0] - pts2d_for_pnp[i][0]), np.float64(x[1] - pts2d_for_pnp[i][1])])
            projpts.append(x)
            if abs(errors[-1][0]) > rep_thresh or abs(errors[-1][1]) > rep_thresh: inliers.append(0)
            else: inliers.append(1)
        a = 0
        for e in errors:
            a = a + abs(e[0]) + abs(e[1])
        avg_err = a/(2*len(errors))
        perc_inliers = sum(inliers)/len(inliers)

        return errors, projpts, avg_err, perc_inliers
    
    
    def get_pair_score(self, unresected_idx, resected_idx, matches, keypoints, 
                       points3d_with_views, img_adjacency):
        """
        Calculates a score for pairing an unresected image with a resected one.
        Higher score is better.
        """
        score = 0
        
        # Check adjacency (basic requirement)
        # Ensure indices are ordered correctly for accessing matches if it's upper/lower triangular
        idx1, idx2 = self.get_idxs_in_correct_order(unresected_idx, resected_idx)
        if not self.images_adjacent(idx1, idx2, img_adjacency): # or use your direct matches check
             return -1 # Invalid pair

        current_matches = matches[idx1, idx2] if idx1==unresected_idx else matches[idx2, idx1] # Adjust if matches isn't symmetric
        if not current_matches: # Or check matches[idx1][idx2] directly if it's an upper triangular matrix
            # If your 'matches' is upper triangular (matches[i][j] where i < j)
            # you'll need to ensure you access it correctly.
            # For example:
            # if unresected_idx < resected_idx:
            #     actual_match_list = matches[unresected_idx][resected_idx]
            # else:
            #     actual_match_list = matches[resected_idx][unresected_idx] # or however you store transposed matches
            # For simplicity, assuming self.matches[i][j] exists and is symmetric or handled by get_idxs_in_correct_order
            # The `get_aligned_kpts` already handles the i<j logic for matches typically.
            
            # Check direct match count using the structure of your `self.matches`
            # This part needs to align with how `self.matches` is structured (symmetric, upper triangular)
            # For an upper triangular `self.matches[i][j]` where i < j:
            u_idx, r_idx = min(unresected_idx, resected_idx), max(unresected_idx, resected_idx)
            if not self.matches[u_idx][r_idx]:
                 return -1


        # Factor 1: Number of matches between this specific pair
        # Again, careful with indexing self.matches
        u_idx, r_idx = min(unresected_idx, resected_idx), max(unresected_idx, resected_idx)
        num_direct_matches = len(self.matches[u_idx, r_idx])
        score += num_direct_matches * 1.0 # Weight for direct matches

        # Factor 2: Number of existing 3D points visible in unresected_idx that are also seen by resected_idx
        # This is essentially what get_correspondences_for_pnp does.
        # For a simpler score, we can just use the number of direct matches from img_adjacency
        # or rely on a more detailed check if performance allows.
        
        # Example: Give a bonus if the resected_idx is "central" or highly connected
        # (This is a placeholder for more sophisticated heuristics)
        # score += len(resected_imgs_seeing_this_resected_idx) * 0.1 
        
        return score

    def next_img_pair_to_grow_reconstruction_scored(self, n_imgs, resected_imgs, unresected_imgs, 
                                                  img_adjacency, matches, keypoints, points3d_with_views):
        """
        Selects the next unresected image and its best resected partner based on a scoring system.
        """
        if not unresected_imgs:
            # This case should be caught by the while loop in main.py
            return None, None, False 

        best_unresected_idx = -1
        best_resected_partner_idx = -1
        highest_score = -1
        chosen_prepend = False # Default, can be adjusted by more complex logic if needed

        # Iterate over all unresected images
        for unres_idx in unresected_imgs:
            # For each unresected image, find its best partner among the resected images
            current_best_partner_for_unres = -1
            current_highest_partner_score = -1

            for res_idx in resected_imgs:
                if unres_idx == res_idx: # Should not happen if lists are managed properly
                    continue
                
                # Ensure the pair is actually connected by matches
                # Your get_idxs_in_correct_order and images_adjacent will be useful
                # Or directly check self.matches using correct indexing
                u_idx_check, r_idx_check = min(unres_idx, res_idx), max(unres_idx, res_idx)
                if not self.matches[u_idx_check,r_idx_check] or len(self.matches[u_idx_check,r_idx_check]) == 0 :
                    continue # No direct matches, skip this resected partner

                # Simplified score: number of matches. More complex scores can be used.
                # This score could also consider the number of shared 3D points for PnP.
                
                # Let's use a simplified score: number of 2D matches between unres_idx and res_idx
                # This requires careful indexing of self.matches, assuming it's upper triangular
                # (i.e., self.matches[i][j] where i < j)
                
                idx_A, idx_B = min(unres_idx, res_idx), max(unres_idx, res_idx)
                score = self.get_pair_score(idx_A, idx_B, matches, keypoints, points3d_with_views, img_adjacency)
                
                # Optionally, add a bonus for resected images that are well-established (e.g., see many 3D points)
                # Or for pairs that offer a good baseline for triangulation.

                if score > current_highest_partner_score:
                    current_highest_partner_score = score
                    current_best_partner_for_unres = res_idx
            
            # If this unresected image found a good partner and its score is the best so far
            if current_best_partner_for_unres != -1 and current_highest_partner_score > highest_score:
                highest_score = current_highest_partner_score
                best_unresected_idx = unres_idx
                best_resected_partner_idx = current_best_partner_for_unres
        
        if best_unresected_idx != -1:
            # Basic prepend logic (can be made more sophisticated)
            # For example, if best_unresected_idx is numerically smaller than the first resected image, prepend.
            if resected_imgs and best_unresected_idx < resected_imgs[0]:
                 chosen_prepend = True
            else:
                 chosen_prepend = False
            print(f"Scored selection: Unresected {best_unresected_idx} with Resected {best_resected_partner_idx} (Score: {highest_score})")
            return best_resected_partner_idx, best_unresected_idx, chosen_prepend
        else:
            # No suitable pair found, this might mean reconstruction cannot proceed with current unresected images.
            # main.py should handle this (e.g., by breaking the loop if unresected_imgs becomes empty).
            print("Warning: No suitable image pair found by next_img_pair_to_grow_reconstruction_scored.")
            if unresected_imgs: # If still images left, try to pick one to allow main.py to remove it if it fails
                # Fallback: pick the first available unresected and a random resected one
                # This is a very basic fallback.
                unres_fallback = unresected_imgs[0]
                res_fallback = random.choice(resected_imgs) if resected_imgs else None
                if res_fallback is not None:
                    print(f"Fallback selection: Unresected {unres_fallback} with Resected {res_fallback}")
                    return res_fallback, unres_fallback, False 
            return None, None, False # Signal failure to find a pair

