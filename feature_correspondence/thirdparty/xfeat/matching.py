import os
import sys
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils import plot_and_save_keypoints, plot_and_save_matches


class XFeatMatcher:
    """
    XFeat-based feature extraction and matching with optional GPU acceleration.
    Produces cv2.KeyPoint and cv2.DMatch outputs compatible with the SfM pipeline.
    """

    def __init__(
        self,
        top_k: int = 4096,
        detection_threshold: float = 0.05,
        min_cossim: float = 0.82,
        min_inlier_matches: int = 20,
        ransac_reprojection_threshold: float = 3.0,
        model_source: str = "auto",
        local_repo_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        input_scale: str = "auto",
    ):
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        self.min_cossim = min_cossim
        self.min_inlier_matches = min_inlier_matches
        self.ransac_reprojection_threshold = ransac_reprojection_threshold
        self.model_source = model_source.lower()
        self.local_repo_path = local_repo_path
        self.weights_path = weights_path

        self.input_scale = input_scale

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            requested = str(device).lower()
            if requested.startswith("cuda") and not torch.cuda.is_available():
                print("XFeat: CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(requested)

        self.xfeat = self._load_xfeat_model()
        self._configure_model_device()

    def _resolve_local_repo(self) -> Optional[str]:
        if self.local_repo_path:
            return self.local_repo_path
        candidate = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "accelerated_features")
        )
        if os.path.isdir(candidate):
            return candidate
        return None

    def _load_xfeat_model(self):
        last_err = None

        if self.model_source in ("auto", "local"):
            local_repo = self._resolve_local_repo()
            if local_repo and os.path.isdir(local_repo):
                if local_repo not in sys.path:
                    sys.path.insert(0, local_repo)
                try:
                    try:
                        from modules.xfeat import XFeat
                    except Exception:
                        from xfeat import XFeat  # type: ignore
                    if self.weights_path:
                        return XFeat(weights=self.weights_path, top_k=self.top_k)
                    return XFeat(top_k=self.top_k)
                except Exception as exc:
                    last_err = exc
            elif self.model_source == "local":
                raise RuntimeError(
                    "XFeat local repo not found. Set XFEAT_LOCAL_REPO or place "
                    "the accelerated_features repo under feature_correspondence/thirdparty/accelerated_features."
                )

        if self.model_source in ("auto", "torchhub"):
            try:
                return torch.hub.load(
                    "verlab/accelerated_features", "XFeat", pretrained=True
                )
            except Exception as exc:
                last_err = exc
            try:
                return torch.hub.load(
                    "verlab/accelerated_features", "xfeat", pretrained=True
                )
            except Exception as exc:
                last_err = exc

        raise RuntimeError(
            "Failed to load XFeat. Ensure accelerated_features is available locally or "
            "torch.hub can access the repository."
        ) from last_err

    def _configure_model_device(self) -> None:
        if hasattr(self.xfeat, "eval"):
            self.xfeat.eval()
        if hasattr(self.xfeat, "to"):
            try:
                self.xfeat.to(self.device)
            except Exception:
                pass
        if hasattr(self.xfeat, "dev"):
            try:
                self.xfeat.dev = self.device
            except Exception:
                pass
        if hasattr(self.xfeat, "net"):
            try:
                self.xfeat.net.to(self.device)
            except Exception:
                pass

    def _normalize_keypoints_output(self, kpts: torch.Tensor) -> np.ndarray:
        if torch.is_tensor(kpts):
            if kpts.dim() == 3:
                kpts = kpts.squeeze(0)
            kpts = kpts.detach().cpu()
            return kpts.numpy()
        return np.asarray(kpts)

    def _normalize_descriptors_output(self, desc: torch.Tensor) -> np.ndarray:
        if torch.is_tensor(desc):
            if desc.dim() == 3:
                desc = desc.squeeze(0)
            desc = desc.detach().cpu()
            return desc.numpy()
        return np.asarray(desc)

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return img
        if not isinstance(img, np.ndarray):
            return img

        if self.input_scale == "none":
            return img

        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0

        img_f = img.astype(np.float32, copy=False)
        if self.input_scale == "auto":
            max_val = float(img_f.max()) if img_f.size else 0.0
            if max_val > 1.5:
                img_f = img_f / 255.0
        elif self.input_scale == "255":
            img_f = img_f / 255.0

        return img_f

    @torch.inference_mode()
    def extract_xfeat_features(
        self, images: List[np.ndarray]
    ) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
        print("\n--- Extracting XFeat Features ---")
        all_keypoints: List[List[cv2.KeyPoint]] = []
        all_descriptors: List[np.ndarray] = []

        for i, img in enumerate(images):
            img = self._prepare_image(img)
            try:
                out = self.xfeat.detectAndCompute(
                    img,
                    top_k=self.top_k,
                    detection_threshold=self.detection_threshold,
                )
            except TypeError:
                out = self.xfeat.detectAndCompute(img, top_k=self.top_k)

            if isinstance(out, list):
                if not out:
                    kpts_np = np.empty((0, 2), dtype=np.float32)
                    desc_np = np.empty((0, 64), dtype=np.float32)
                else:
                    out = out[0]
                    kpts_np = self._normalize_keypoints_output(out["keypoints"])
                    desc_np = self._normalize_descriptors_output(out["descriptors"])
            else:
                kpts_np = self._normalize_keypoints_output(out["keypoints"])
                desc_np = self._normalize_descriptors_output(out["descriptors"])

            if kpts_np.size == 0:
                keypoints = []
            else:
                keypoints = [
                    cv2.KeyPoint(float(x), float(y), 1)
                    for x, y in kpts_np.reshape(-1, 2)
                ]

            desc_np = desc_np.astype(np.float32, copy=False)
            all_keypoints.append(keypoints)
            all_descriptors.append(desc_np)
            print(f"  Extracted {len(keypoints)} XFeat features from image #{i}")

        return all_keypoints, all_descriptors

    @torch.inference_mode()
    def match_descriptors(
        self, desc1: np.ndarray, desc2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
            )

        t1 = torch.from_numpy(desc1).to(self.device)
        t2 = torch.from_numpy(desc2).to(self.device)
        if t1.dtype != torch.float32:
            t1 = t1.float()
        if t2.dtype != torch.float32:
            t2 = t2.float()

        t1 = F.normalize(t1, dim=1)
        t2 = F.normalize(t2, dim=1)

        sim = t1 @ t2.t()
        match12 = torch.argmax(sim, dim=1)
        match21 = torch.argmax(sim, dim=0)

        idx0 = torch.arange(len(match12), device=self.device)
        mutual = match21[match12] == idx0

        if self.min_cossim is not None and self.min_cossim > 0:
            max_sim = sim.max(dim=1).values
            keep = mutual & (max_sim > self.min_cossim)
        else:
            keep = mutual

        idx0 = idx0[keep]
        idx1 = match12[keep]
        scores = sim[idx0, idx1]

        return (
            idx0.detach().cpu().numpy(),
            idx1.detach().cpu().numpy(),
            scores.detach().cpu().numpy(),
        )

    def find_raw_matches(
        self, image_descriptors: List[np.ndarray]
    ) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
        print("\n--- Performing XFeat Matching ---")
        num_images = len(image_descriptors)
        raw_matches_map: Dict[Tuple[int, int], List[cv2.DMatch]] = {}
        pair_window_env = os.getenv("XFEAT_PAIR_WINDOW") or os.getenv("PAIR_WINDOW")
        max_matches_env = os.getenv("XFEAT_MAX_MATCHES_PER_PAIR")
        pair_window = int(pair_window_env) if pair_window_env else 0
        max_matches = int(max_matches_env) if max_matches_env else 0

        for i in range(num_images):
            j_start = i + 1
            j_end = num_images if pair_window <= 0 else min(num_images, i + 1 + pair_window)
            for j in range(j_start, j_end):
                desc1 = image_descriptors[i]
                desc2 = image_descriptors[j]
                if desc1.size == 0 or desc2.size == 0:
                    print(
                        f"  Skipping match between image #{i} and #{j}: one or both have no descriptors."
                    )
                    continue

                idxs0, idxs1, scores = self.match_descriptors(desc1, desc2)
                if max_matches > 0 and len(scores) > max_matches:
                    top_idx = np.argsort(-scores)[:max_matches]
                    idxs0 = idxs0[top_idx]
                    idxs1 = idxs1[top_idx]
                    scores = scores[top_idx]
                matches = []
                for k in range(len(idxs0)):
                    distance = float(1.0 - scores[k])
                    matches.append(
                        cv2.DMatch(
                            _queryIdx=int(idxs0[k]),
                            _trainIdx=int(idxs1[k]),
                            _distance=distance,
                        )
                    )

                raw_matches_map[(i, j)] = matches
                print(
                    f"  Image #{i} and Image #{j}: {len(matches)} raw matches (MNN)"
                )

        return raw_matches_map

    def filter_matches_geometric_consistency(
        self,
        raw_matches_map: Dict[Tuple[int, int], List[cv2.DMatch]],
        all_image_keypoints: List[List[cv2.KeyPoint]],
    ) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
        print("\n--- Filtering Matches for Geometric Consistency (RANSAC) ---")
        filtered_matches_map: Dict[Tuple[int, int], List[cv2.DMatch]] = {}

        for (idx1, idx2), matches_list in raw_matches_map.items():
            filtered_matches_map[(idx1, idx2)] = []
            if len(matches_list) < self.min_inlier_matches:
                print(
                    f"  Skipping RANSAC for Image #{idx1} and #{idx2}: too few raw matches ({len(matches_list)})"
                )
                continue

            src_pts = np.float32(
                [all_image_keypoints[idx1][m.queryIdx].pt for m in matches_list]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [all_image_keypoints[idx2][m.trainIdx].pt for m in matches_list]
            ).reshape(-1, 1, 2)

            fundamental_matrix, mask = cv2.findFundamentalMat(
                src_pts,
                dst_pts,
                cv2.FM_RANSAC,
                self.ransac_reprojection_threshold,
                0.99,
            )

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
        print("\n--- Building Image Connectivity Graph ---")
        adjacency_matrix = np.zeros((num_images, num_images), dtype=int)
        connected_image_pairs: List[Tuple[int, int]] = []

        for (idx1, idx2), matches in filtered_matches.items():
            if matches:
                adjacency_matrix[idx1, idx2] = 1
                adjacency_matrix[idx2, idx1] = 1
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
        output_plots_base_dir = os.path.join(os.getcwd(), "output_plots", dataset_name)
        features_out_dir = os.path.join(output_plots_base_dir, "features")
        matches_out_dir = os.path.join(output_plots_base_dir, "feature_matches")
        os.makedirs(features_out_dir, exist_ok=True)
        os.makedirs(matches_out_dir, exist_ok=True)
        print(f"Saving feature plots to: {features_out_dir}")
        print(f"Saving match plots to: {matches_out_dir}")

        for i in range(num_images):
            img_for_plot = (
                images_color_for_plotting[i] if images_color_for_plotting else images[i]
            )
            plot_and_save_keypoints(
                img_for_plot,
                all_keypoints[i],
                os.path.join(features_out_dir, f"features_img_{i:03d}.png"),
                title=f"Detected {len(all_keypoints[i])} XFeat Features in Image {i}",
                show_plot=False,
            )

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
                        filtered_matches[i, j],
                        os.path.join(
                            matches_out_dir, f"matches_{i:03d}_vs_{j:03d}.png"
                        ),
                        title=f"Filtered Matches {len(filtered_matches[i,j])} between Image {i} and Image {j}",
                        show_plot=False,
                    )

    def process_images(
        self,
        images: List[np.ndarray],
        save_plot=False,
        dataset_name="",
        images_color_for_plotting: List[np.ndarray] = [],
    ) -> Dict[str, any]:
        if not images:
            print("No images provided for processing.")
            return {}

        all_keypoints, all_descriptors = self.extract_xfeat_features(images)

        raw_matches = self.find_raw_matches(all_descriptors)

        filtered_matches = self.filter_matches_geometric_consistency(
            raw_matches, all_keypoints
        )

        num_images = len(images)
        adjacency_matrix, connected_pairs = self.build_connectivity_graph(
            num_images, filtered_matches
        )

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
