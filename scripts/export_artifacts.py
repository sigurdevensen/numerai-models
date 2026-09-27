"""Writes trained model artifacts (weights, feature set, holdout metrics) to disk.

Called from Mojo (src/train.mojo) after the holdout validation gate passes.
"""
import json
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def save(weights: np.ndarray, bias: float, feature_set, metrics: dict, mean: np.ndarray, std: np.ndarray):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.save(ARTIFACTS_DIR / "weights.npy", np.asarray(weights, dtype=np.float32))
    np.save(ARTIFACTS_DIR / "feature_mean.npy", np.asarray(mean, dtype=np.float32))
    np.save(ARTIFACTS_DIR / "feature_std.npy", np.asarray(std, dtype=np.float32))
    meta = {
        "bias": float(bias),
        "feature_set": list(feature_set),
        "holdout_metrics": {k: v for k, v in dict(metrics).items()},
    }
    with open(ARTIFACTS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote weights/feature_mean/feature_std.npy and model_meta.json to {ARTIFACTS_DIR}")
