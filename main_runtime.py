import os
import sys

from feature_correspondence import feature_correspondence_runtime as _fcr

# Ensure main.py imports the runtime (writable) implementation without editing root-owned files.
sys.modules["feature_correspondence.feature_correspondence"] = _fcr

import main as _main

if os.getenv("FAST_MODE") == "1":
    # Feature extraction / matching
    os.environ.setdefault("XFEAT_TOP_K", "2048")
    os.environ.setdefault("XFEAT_PAIR_WINDOW", "3")
    os.environ.setdefault("XFEAT_MAX_MATCHES_PER_PAIR", "3000")

    # Bundle adjustment speed/quality tradeoffs
    os.environ.setdefault("BA_DEVICE", "cuda")
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
_main.LOCAL_BA_N_ITER_TORCH = _env_int("LOCAL_BA_N_ITER_TORCH", 400)
_main.LOCAL_BA_LEARNING_RATE_TORCH = _env_float("LOCAL_BA_LEARNING_RATE_TORCH", 5e-4)
_main.GLOBAL_BA_N_ITER_TORCH = _env_int("GLOBAL_BA_N_ITER_TORCH", 1000)
_main.GLOBAL_BA_LEARNING_RATE_TORCH = _env_float(
    "GLOBAL_BA_LEARNING_RATE_TORCH", 5e-5
)

if os.getenv("SKIP_VIS") == "1":
    _main.visualize_reconstruction = lambda *args, **kwargs: None


if __name__ == "__main__":
    _main.main()
