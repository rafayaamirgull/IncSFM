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
        self.batch_size = max(1, int(os.getenv("XFEAT_BATCH_SIZE", "8")))
        amp_env = os.getenv("XFEAT_USE_AMP", "1")
        self.use_amp = amp_env == "1"
        self.dynamic_batch = os.getenv("XFEAT_DYNAMIC_BATCH", "1") == "1"
        self.batch_size_large = max(
            self.batch_size, int(os.getenv("XFEAT_BATCH_SIZE_LARGE", "16"))
        )
        self.batch_size_very_large = max(
            self.batch_size_large, int(os.getenv("XFEAT_BATCH_SIZE_VERY_LARGE", "24"))
        )

        self.parallel_mode = os.getenv("XFEAT_PARALLEL_MODE", "auto").lower()
        self.parallel_min_images = int(os.getenv("XFEAT_PARALLEL_MIN_IMAGES", "60"))
        self.parallel_min_total_features = int(
            os.getenv("XFEAT_PARALLEL_MIN_TOTAL_FEATURES", "180000")
        )
        self.parallel_min_pairs = int(os.getenv("XFEAT_PARALLEL_MIN_PAIRS", "120"))
        self.parallel_min_pair_work = int(
            os.getenv("XFEAT_PARALLEL_MIN_PAIR_WORK", "8000000000")
        )
        self.cuda_streams = max(1, int(os.getenv("XFEAT_CUDA_STREAMS", "4")))
        self.cuda_pair_chunk = max(1, int(os.getenv("XFEAT_PAIR_CHUNK", "32")))

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
        if self.device.type != "cuda":
            self.use_amp = False

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

    def _run_detect_and_compute(self, image_input):
        if self.use_amp and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                try:
                    return self.xfeat.detectAndCompute(
                        image_input,
                        top_k=self.top_k,
                        detection_threshold=self.detection_threshold,
                    )
                except TypeError:
                    return self.xfeat.detectAndCompute(
                        image_input,
                        top_k=self.top_k,
                    )

        try:
            return self.xfeat.detectAndCompute(
                image_input,
                top_k=self.top_k,
                detection_threshold=self.detection_threshold,
            )
        except TypeError:
            return self.xfeat.detectAndCompute(
                image_input,
                top_k=self.top_k,
            )

    def _parse_model_output(
        self, out_obj
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        if isinstance(out_obj, list):
            if not out_obj:
                return [], np.empty((0, 64), dtype=np.float32)
            out_obj = out_obj[0]

        kpts_np = self._normalize_keypoints_output(out_obj["keypoints"])
        desc_np = self._normalize_descriptors_output(out_obj["descriptors"])

        if kpts_np.size == 0:
            keypoints = []
        else:
            keypoints = [
                cv2.KeyPoint(float(x), float(y), 1)
                for x, y in kpts_np.reshape(-1, 2)
            ]

        return keypoints, desc_np.astype(np.float32, copy=False)

    def _extract_single_image(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        image_prepped = self._prepare_image(image)
        out = self._run_detect_and_compute(image_prepped)
        return self._parse_model_output(out)

    def _effective_batch_size(self, n_images: int, images: List[np.ndarray]) -> int:
        if not self.dynamic_batch:
            return self.batch_size

        total_pixels = 0
        for img in images:
            if isinstance(img, np.ndarray) and img.ndim >= 2:
                total_pixels += int(img.shape[0] * img.shape[1])

        if n_images >= 120 or total_pixels >= 120 * 640 * 480:
            return self.batch_size_very_large
        if n_images >= self.parallel_min_images or total_pixels >= 60 * 640 * 480:
            return self.batch_size_large
        return self.batch_size

    @torch.inference_mode()
    def extract_xfeat_features(
        self, images: List[np.ndarray]
    ) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
        print("\n--- Extracting XFeat Features ---")
        n_images = len(images)
        all_keypoints: List[List[cv2.KeyPoint]] = [[] for _ in range(n_images)]
        all_descriptors: List[np.ndarray] = [
            np.empty((0, 64), dtype=np.float32) for _ in range(n_images)
        ]

        effective_batch_size = self._effective_batch_size(n_images, images)
        if effective_batch_size != self.batch_size:
            print(
                f"  XFeat dynamic extraction batch: {self.batch_size} -> {effective_batch_size}"
            )

        if effective_batch_size <= 1:
            for i, img in enumerate(images):
                keypoints, descriptors = self._extract_single_image(img)
                all_keypoints[i] = keypoints
                all_descriptors[i] = descriptors
                print(f"  Extracted {len(keypoints)} XFeat features from image #{i}")
            return all_keypoints, all_descriptors

        buckets: Dict[Tuple[int, ...], List[int]] = {}
        fallback_indices: List[int] = []

        for idx, img in enumerate(images):
            if not isinstance(img, np.ndarray):
                fallback_indices.append(idx)
                continue
            buckets.setdefault(tuple(img.shape), []).append(idx)

        for idx in fallback_indices:
            keypoints, descriptors = self._extract_single_image(images[idx])
            all_keypoints[idx] = keypoints
            all_descriptors[idx] = descriptors

        for _, indices in buckets.items():
            for start in range(0, len(indices), effective_batch_size):
                chunk = indices[start : start + effective_batch_size]
                chunk_imgs_raw = [images[i] for i in chunk]

                try:
                    chunk_imgs = [self._prepare_image(img) for img in chunk_imgs_raw]
                    if any(not isinstance(img, np.ndarray) for img in chunk_imgs):
                        raise RuntimeError(
                            "Non-numpy image found in XFeat batch preparation."
                        )
                    stacked = np.stack(chunk_imgs, axis=0)
                    batch_tensor = torch.from_numpy(stacked)
                    if batch_tensor.dim() == 3:
                        batch_tensor = batch_tensor.unsqueeze(1)
                    elif batch_tensor.dim() == 4:
                        batch_tensor = batch_tensor.permute(0, 3, 1, 2)
                    else:
                        raise ValueError(
                            f"Unsupported batch tensor shape: {tuple(batch_tensor.shape)}"
                        )
                    batch_tensor = batch_tensor.float()
                    out_batch = self._run_detect_and_compute(batch_tensor)
                    if not isinstance(out_batch, list) or len(out_batch) != len(chunk):
                        raise RuntimeError(
                            "Unexpected batched output from XFeat detectAndCompute."
                        )

                    for local_idx, out_obj in enumerate(out_batch):
                        keypoints, descriptors = self._parse_model_output(out_obj)
                        image_idx = chunk[local_idx]
                        all_keypoints[image_idx] = keypoints
                        all_descriptors[image_idx] = descriptors
                except Exception as exc:
                    print(
                        f"  XFeat batch extraction fallback on {len(chunk)} images ({exc})."
                    )
                    for image_idx in chunk:
                        keypoints, descriptors = self._extract_single_image(images[image_idx])
                        all_keypoints[image_idx] = keypoints
                        all_descriptors[image_idx] = descriptors

        for i in range(n_images):
            print(f"  Extracted {len(all_keypoints[i])} XFeat features from image #{i}")

        return all_keypoints, all_descriptors

    @torch.inference_mode()
    def _prepare_descriptor_tensor(self, desc: np.ndarray) -> Optional[torch.Tensor]:
        if desc is None or desc.size == 0:
            return None
        t = torch.from_numpy(desc).to(self.device)
        if t.dtype != torch.float32:
            t = t.float()
        t = F.normalize(t, dim=1)
        if self.use_amp and self.device.type == "cuda":
            t = t.half()
        return t

    @torch.inference_mode()
    def _match_descriptor_tensors_torch(
        self, t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if t1 is None or t2 is None or t1.numel() == 0 or t2.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.float32, device=self.device),
            )

        sim = t1 @ t2.t()
        if sim.dtype != torch.float32:
            sim = sim.float()

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

        return idx0, idx1, scores

    @torch.inference_mode()
    def _match_descriptor_tensors(
        self, t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx0, idx1, scores = self._match_descriptor_tensors_torch(t1, t2)

        return (
            idx0.cpu().numpy(),
            idx1.cpu().numpy(),
            scores.cpu().numpy(),
        )

    @torch.inference_mode()
    def match_descriptors(
        self, desc1: np.ndarray, desc2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._match_descriptor_tensors(
            self._prepare_descriptor_tensor(desc1),
            self._prepare_descriptor_tensor(desc2),
        )

    def _build_candidate_pairs(
        self,
        image_descriptors: List[np.ndarray],
        pair_window: int,
        global_topk: int,
        global_min_gap: int,
        global_min_sim: float,
    ) -> List[Tuple[int, int]]:
        num_images = len(image_descriptors)
        pair_set = set()

        for i in range(num_images):
            j_start = i + 1
            j_end = (
                num_images
                if pair_window <= 0
                else min(num_images, i + 1 + pair_window)
            )
            for j in range(j_start, j_end):
                pair_set.add((i, j))

        if global_topk <= 0 or num_images <= 1:
            return sorted(pair_set)

        desc_dim = None
        for desc in image_descriptors:
            if desc is not None and desc.size > 0:
                desc_dim = desc.shape[1]
                break
        if desc_dim is None:
            return sorted(pair_set)

        global_desc = np.zeros((num_images, desc_dim), dtype=np.float32)
        valid = np.zeros(num_images, dtype=bool)
        for i, desc in enumerate(image_descriptors):
            if desc is None or desc.size == 0:
                continue
            pooled = desc.mean(axis=0).astype(np.float32, copy=False)
            norm = np.linalg.norm(pooled)
            if norm <= 1e-8:
                continue
            global_desc[i] = pooled / norm
            valid[i] = True

        if valid.sum() <= 1:
            return sorted(pair_set)

        similarity = global_desc @ global_desc.T
        np.fill_diagonal(similarity, -np.inf)
        nonlocal_start_gap = max(global_min_gap, pair_window + 1)

        for i in range(num_images):
            if not valid[i]:
                continue
            row = similarity[i].copy()
            if nonlocal_start_gap > 0:
                left = max(0, i - nonlocal_start_gap + 1)
                right = min(num_images, i + nonlocal_start_gap)
                row[left:right] = -np.inf

            ranked = np.argsort(-row)
            added = 0
            for j in ranked:
                score = float(row[j])
                if not np.isfinite(score):
                    break
                if score < global_min_sim:
                    break
                pair_set.add((min(i, int(j)), max(i, int(j))))
                added += 1
                if added >= global_topk:
                    break

        return sorted(pair_set)

    def _build_cv2_matches(
        self, idxs0: np.ndarray, idxs1: np.ndarray, scores: np.ndarray
    ) -> List[cv2.DMatch]:
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
        return matches

    def _should_enable_cuda_parallel(
        self, image_descriptors: List[np.ndarray], candidate_pairs: List[Tuple[int, int]]
    ) -> bool:
        if self.device.type != "cuda" or self.cuda_streams <= 1 or not candidate_pairs:
            return False

        mode = self.parallel_mode
        if mode == "off":
            return False
        if mode == "on":
            return True

        total_features = 0
        feature_counts: List[int] = []
        for desc in image_descriptors:
            if desc is not None and desc.size > 0:
                n_feat = int(desc.shape[0])
                total_features += n_feat
                feature_counts.append(n_feat)
            else:
                feature_counts.append(0)

        if (
            len(image_descriptors) >= self.parallel_min_images
            or total_features >= self.parallel_min_total_features
        ):
            return True

        if len(candidate_pairs) < self.parallel_min_pairs:
            return False

        if self.parallel_min_pair_work > 0:
            pair_work = 0
            for i, j in candidate_pairs:
                ni = feature_counts[i]
                nj = feature_counts[j]
                if ni <= 0 or nj <= 0:
                    continue
                pair_work += ni * nj
                if pair_work >= self.parallel_min_pair_work:
                    return True

        return False

    @torch.inference_mode()
    def _find_raw_matches_cuda_parallel(
        self,
        image_descriptors: List[np.ndarray],
        candidate_pairs: List[Tuple[int, int]],
        desc_tensors: List[Optional[torch.Tensor]],
        max_matches: int,
    ) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
        raw_matches_map: Dict[Tuple[int, int], List[cv2.DMatch]] = {}
        streams = [torch.cuda.Stream(device=self.device) for _ in range(self.cuda_streams)]
        print(
            f"  Using CUDA parallel matching: {self.cuda_streams} streams, chunk={self.cuda_pair_chunk}"
        )

        for start in range(0, len(candidate_pairs), self.cuda_pair_chunk):
            chunk = candidate_pairs[start : start + self.cuda_pair_chunk]

            # Keep in-flight work bounded by stream count to avoid GPU memory spikes.
            for wave_start in range(0, len(chunk), self.cuda_streams):
                wave = chunk[wave_start : wave_start + self.cuda_streams]
                launched = []
                skipped = []

                for stream_idx, (i, j) in enumerate(wave):
                    desc1 = image_descriptors[i]
                    desc2 = image_descriptors[j]
                    if desc1.size == 0 or desc2.size == 0:
                        skipped.append((i, j))
                        continue

                    stream = streams[stream_idx]
                    with torch.cuda.stream(stream):
                        idx0_t, idx1_t, scores_t = self._match_descriptor_tensors_torch(
                            desc_tensors[i], desc_tensors[j]
                        )
                        if max_matches > 0 and scores_t.numel() > max_matches:
                            top_idx = torch.topk(
                                scores_t, k=max_matches, largest=True, sorted=False
                            ).indices
                            idx0_t = idx0_t[top_idx]
                            idx1_t = idx1_t[top_idx]
                            scores_t = scores_t[top_idx]
                    launched.append((i, j, idx0_t, idx1_t, scores_t))

                torch.cuda.synchronize(self.device)

                for i, j in skipped:
                    print(
                        f"  Skipping match between image #{i} and #{j}: one or both have no descriptors."
                    )

                for i, j, idx0_t, idx1_t, scores_t in launched:
                    idxs0 = idx0_t.cpu().numpy()
                    idxs1 = idx1_t.cpu().numpy()
                    scores = scores_t.cpu().numpy()
                    matches = self._build_cv2_matches(idxs0, idxs1, scores)
                    raw_matches_map[(i, j)] = matches
                    print(
                        f"  Image #{i} and Image #{j}: {len(matches)} raw matches (MNN)"
                    )

        return raw_matches_map

    def find_raw_matches(
        self, image_descriptors: List[np.ndarray]
    ) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
        print("\n--- Performing XFeat Matching ---")
        raw_matches_map: Dict[Tuple[int, int], List[cv2.DMatch]] = {}
        pair_window_env = os.getenv("XFEAT_PAIR_WINDOW") or os.getenv("PAIR_WINDOW")
        max_matches_env = os.getenv("XFEAT_MAX_MATCHES_PER_PAIR")
        global_topk_env = os.getenv("XFEAT_GLOBAL_TOPK")
        global_min_gap_env = os.getenv("XFEAT_GLOBAL_MIN_GAP")
        global_min_sim_env = os.getenv("XFEAT_GLOBAL_MIN_SIM")
        pair_window = int(pair_window_env) if pair_window_env else 0
        max_matches = int(max_matches_env) if max_matches_env else 0
        global_topk = int(global_topk_env) if global_topk_env else 0
        global_min_gap = int(global_min_gap_env) if global_min_gap_env else 0
        global_min_sim = float(global_min_sim_env) if global_min_sim_env else 0.15

        candidate_pairs = self._build_candidate_pairs(
            image_descriptors=image_descriptors,
            pair_window=pair_window,
            global_topk=global_topk,
            global_min_gap=global_min_gap,
            global_min_sim=global_min_sim,
        )
        print(f"  Candidate pairs for matching: {len(candidate_pairs)}")

        desc_tensors = [
            self._prepare_descriptor_tensor(desc) for desc in image_descriptors
        ]

        if self._should_enable_cuda_parallel(image_descriptors, candidate_pairs):
            return self._find_raw_matches_cuda_parallel(
                image_descriptors=image_descriptors,
                candidate_pairs=candidate_pairs,
                desc_tensors=desc_tensors,
                max_matches=max_matches,
            )

        for i, j in candidate_pairs:
            desc1 = image_descriptors[i]
            desc2 = image_descriptors[j]
            if desc1.size == 0 or desc2.size == 0:
                print(
                    f"  Skipping match between image #{i} and #{j}: one or both have no descriptors."
                )
                continue

            idxs0, idxs1, scores = self._match_descriptor_tensors(
                desc_tensors[i], desc_tensors[j]
            )
            if max_matches > 0 and len(scores) > max_matches:
                top_idx = np.argsort(-scores)[:max_matches]
                idxs0 = idxs0[top_idx]
                idxs1 = idxs1[top_idx]
                scores = scores[top_idx]
            matches = self._build_cv2_matches(idxs0, idxs1, scores)

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
