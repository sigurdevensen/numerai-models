"""Loads Numerai training data and prepares train/holdout arrays for the Mojo training pipeline.

Called from Mojo (src/train.mojo) via Python interop, and usable standalone from Python.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_VERSION = "v5.3"


def load_feature_set(feature_set_name="small"):
    with open(DATA_DIR / "features.json") as f:
        meta = json.load(f)
    return meta["feature_sets"][feature_set_name]


def _to_arrays(df, feature_set):
    # Numerai features are integers in [0, 4]; scale to [0, 1] for gradient descent.
    # .copy() forces a writable, contiguous buffer -- pandas' CoW mode can hand back
    # read-only arrays from to_numpy(), which Mojo's numpy interop rejects.
    X = np.ascontiguousarray(df[feature_set].to_numpy(dtype=np.float32) / 4.0).copy()
    y = np.ascontiguousarray(df["target"].to_numpy(dtype=np.float32)).copy()
    eras = np.ascontiguousarray(df["era"].to_numpy(dtype=np.int32)).copy()
    return X, y, eras


def load_split(feature_set_name="small", holdout_eras=60, embargo_eras=13):
    """Returns (feature_set, fit_X, fit_y, holdout_X, holdout_y, holdout_eras, mean, std).

    Mirrors legacy/lightgbm-baseline/train_and_submit.ipynb's holdout scheme: hold out
    the most recent `holdout_eras`, embargo the `embargo_eras` immediately before them
    (the target looks ~13 weekly eras into the future, so rows right before the holdout
    would otherwise leak into it), and fit on everything earlier.

    Features are standardized (zero mean, unit variance) using stats from the fit
    split only -- gradient descent on Numerai's raw [0, 1]-scaled features is poorly
    conditioned (they're all non-negative, so the loss surface is badly skewed) and
    diverges at any learning rate worth using. `mean`/`std` are returned so
    scripts/export_artifacts.py can save them: predict() at submission time must
    apply the exact same transform to live features.
    """
    feature_set = load_feature_set(feature_set_name)
    train_path = DATA_DIR / DATA_VERSION / "train.parquet"
    df = pd.read_parquet(train_path, columns=["era", "target"] + feature_set)
    df = df.dropna(subset=["target"])

    eras = sorted(df["era"].unique(), key=int)
    holdout_era_set = set(eras[-holdout_eras:])
    embargo_era_set = set(eras[-(holdout_eras + embargo_eras) : -holdout_eras])
    fit_era_set = set(eras) - holdout_era_set - embargo_era_set

    fit_df = df[df["era"].isin(fit_era_set)]
    holdout_df = df[df["era"].isin(holdout_era_set)]

    fit_X, fit_y, _ = _to_arrays(fit_df, feature_set)
    holdout_X, holdout_y, holdout_eras_arr = _to_arrays(holdout_df, feature_set)

    mean = fit_X.mean(axis=0).astype(np.float32)
    std = fit_X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    fit_X = np.ascontiguousarray((fit_X - mean) / std).copy()
    holdout_X = np.ascontiguousarray((holdout_X - mean) / std).copy()

    print(
        f"fit eras: {len(fit_era_set)} ({fit_X.shape[0]} rows), "
        f"embargoed eras: {len(embargo_era_set)}, "
        f"holdout eras: {len(holdout_era_set)} ({holdout_X.shape[0]} rows), "
        f"features: {len(feature_set)}"
    )
    return feature_set, fit_X, fit_y, holdout_X, holdout_y, holdout_eras_arr, mean, std


if __name__ == "__main__":
    import time

    t0 = time.time()
    feature_set, fit_X, fit_y, hold_X, hold_y, hold_eras, mean, std = load_split()
    print(f"loaded in {time.time() - t0:.1f}s")
    print(fit_X.shape, fit_X.dtype, fit_y.shape, hold_X.shape, hold_y.shape)
