"""Simple XGBoost baseline for Numerai."""
import sys
from pathlib import Path

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from numerai_data import load_split
from validate import per_era_corr, per_era_corr_series

def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    # Must mirror scripts/numerai_data.py's preprocessing exactly: scale
    # raw [0, 4] features to [0, 1], then standardize with the fit-time
    # mean/std baked in at export time.
    x = live_features[feature_set].to_numpy(dtype=np.float32) / 4.0
    x = (x - mean) / std
    preds = model.predict(x)
    return pd.Series(preds, index=live_features.index).to_frame("prediction")

feature_set, fit_X, fit_y, holdout_X, holdout_y, holdout_eras, mean, std = load_split(
    feature_set_name="small"
)

model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    colsample_bytree=0.1,
)
model.fit(fit_X, fit_y)

predictions = model.predict(holdout_X)
metrics = per_era_corr(predictions, holdout_y, holdout_eras, min_mean=0.0, min_sharpe=0.3)
print(metrics)

ARTIFACTS_DIR.mkdir(exist_ok=True)
model.save_model(ARTIFACTS_DIR / "xgboost_model.json")
print(f"saved model to {ARTIFACTS_DIR / 'xgboost_model.json'}")

if metrics["passed"]:
    pkl_path = ARTIFACTS_DIR / "xgboost_model.pkl"
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(predict, f)
    print(f"wrote {pkl_path.resolve()} ({pkl_path.stat().st_size / 1e3:.1f} KB)")
else:
    print(
        f"holdout check failed (mean CORR {metrics['mean']:.4f} <= 0.0 or "
        f"sharpe {metrics['sharpe']:.4f} <= 0.3) -- skipping submission pickle."
    )

corr_series = per_era_corr_series(predictions, holdout_y, holdout_eras)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
corr_series.plot(
    title="Validation CORR",
    kind="bar",
    ax=axes[0],
    xticks=[],
    legend=False,
    snap=False,
)
corr_series.cumsum().plot(
    title="Cumulative Validation CORR",
    kind="line",
    ax=axes[1],
    legend=False,
)
plt.tight_layout()
plt.show()