"""Packages the trained Mojo model for Numerai Compute and (optionally) uploads it.

Numerai Compute runs a pickled Python `predict(live_features) -> DataFrame`
callable -- it cannot run a Mojo binary directly. This wraps the weights that
src/train.mojo exported to artifacts/ in a thin numpy predict() function that
reproduces the trained linear model exactly, then cloudpickles it.

Usage:
    pixi run python scripts/submit.py package                 # writes artifacts/model.pkl
    pixi run python scripts/submit.py upload --model-id <id>  # package + upload to Numerai Compute

Uploading requires NUMERAI_PUBLIC_ID / NUMERAI_SECRET_KEY env vars (an API key
with the upload_submission scope, from numer.ai/account) and is NOT run
automatically -- it's a real submission to your live Numerai model slot.
"""
import argparse
import json
import os
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_VERSION = "v5.3"


def package():
    weights = np.load(ARTIFACTS_DIR / "weights.npy")
    mean = np.load(ARTIFACTS_DIR / "feature_mean.npy")
    std = np.load(ARTIFACTS_DIR / "feature_std.npy")
    with open(ARTIFACTS_DIR / "model_meta.json") as f:
        meta = json.load(f)
    bias = meta["bias"]
    feature_set = meta["feature_set"]

    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        # Must mirror scripts/numerai_data.py's preprocessing exactly: scale
        # raw [0, 4] features to [0, 1], then standardize with the fit-time
        # mean/std baked in at export time.
        x = live_features[feature_set].to_numpy(dtype=np.float32) / 4.0
        x = (x - mean) / std
        predictions = x @ weights + bias
        return pd.Series(predictions, index=live_features.index).to_frame("prediction")

    pkl_path = ARTIFACTS_DIR / "model.pkl"
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(predict, f)
    print(f"wrote {pkl_path.resolve()} ({pkl_path.stat().st_size / 1e3:.1f} KB)")
    return pkl_path


def upload(model_id: str, pkl_path: Path, docker_image: str):
    from numerapi import NumerAPI

    napi = NumerAPI(
        public_id=os.environ["NUMERAI_PUBLIC_ID"],
        secret_key=os.environ["NUMERAI_SECRET_KEY"],
    )
    upload_id = napi.model_upload(
        str(pkl_path),
        model_id=model_id,
        docker_image=docker_image,
        data_version=DATA_VERSION,
    )
    print(f"uploaded, compute pickle id: {upload_id}")
    print("Numerai validates the pickle against live data before it goes live --")
    print(f"check status at numer.ai or with NumerAPI().model_upload_data_versions().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("package")
    upload_parser = sub.add_parser("upload")
    upload_parser.add_argument("--model-id", required=True, help="Numerai model UUID")
    upload_parser.add_argument(
        "--docker-image",
        default="Python 3.11",
        help="Must match the Python version used to train/pickle (see napi.model_upload_docker_images())",
    )
    args = parser.parse_args()

    pkl_path = package()
    if args.command == "upload":
        upload(args.model_id, pkl_path, args.docker_image)
