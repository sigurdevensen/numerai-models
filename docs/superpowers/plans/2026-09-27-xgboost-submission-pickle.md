# XGBoost Submission Pickle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/xgboost-model/main.py` save its trained model as a cloudpickled `predict()`
callable (`artifacts/xgboost_model.pkl`) that Numerai Compute can run directly, gated on the
same holdout bar the Mojo pipeline uses.

**Architecture:** Single-file change. After scoring holdout predictions, gate on
`per_era_corr(..., min_mean=0.0, min_sharpe=0.3)`. If it passes, build a `predict(live_features)
-> DataFrame` closure over the already-in-memory `model`, `mean`, `std`, `feature_set` that
reproduces `numerai_data.py`'s preprocessing exactly, and `cloudpickle` it straight to
`artifacts/xgboost_model.pkl` — no intermediate files, no reload step (this script never crosses
a Mojo/Python boundary the way `scripts/submit.py` does, so it doesn't need one).

**Tech Stack:** Python (system Python, not pixi — pixi is only for Mojo in this repo),
`cloudpickle`, `pandas`, `numpy`, `xgboost`.

**Reference spec:** `docs/superpowers/specs/2026-09-27-xgboost-submission-pickle-design.md`

---

### Task 1: Add the holdout gate and submission pickle to `src/xgboost-model/main.py`

**Files:**
- Modify: `src/xgboost-model/main.py`

Current file (for reference — this is the exact content being changed):

```python
"""Simple XGBoost baseline for Numerai."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from numerai_data import load_split
from validate import per_era_corr, per_era_corr_series


def main():
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
    metrics = per_era_corr(predictions, holdout_y, holdout_eras)
    print(metrics)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model.save_model(ARTIFACTS_DIR / "xgboost_model.json")
    print(f"saved model to {ARTIFACTS_DIR / 'xgboost_model.json'}")

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


if __name__ == "__main__":
    main()
```

- [ ] **Step 1: Add the new imports**

`main.py` needs `numpy`, `pandas`, and `cloudpickle` for the submission closure, none of which
are currently imported. Change the top of the file from:

```python
"""Simple XGBoost baseline for Numerai."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from xgboost import XGBRegressor
```

to:

```python
"""Simple XGBoost baseline for Numerai."""
import sys
from pathlib import Path

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
```

- [ ] **Step 2: Gate the holdout score and add the submission pickle**

Change:

```python
    predictions = model.predict(holdout_X)
    metrics = per_era_corr(predictions, holdout_y, holdout_eras)
    print(metrics)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model.save_model(ARTIFACTS_DIR / "xgboost_model.json")
    print(f"saved model to {ARTIFACTS_DIR / 'xgboost_model.json'}")

    corr_series = per_era_corr_series(predictions, holdout_y, holdout_eras)
```

to:

```python
    predictions = model.predict(holdout_X)
    metrics = per_era_corr(predictions, holdout_y, holdout_eras, min_mean=0.0, min_sharpe=0.3)
    print(metrics)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model.save_model(ARTIFACTS_DIR / "xgboost_model.json")
    print(f"saved model to {ARTIFACTS_DIR / 'xgboost_model.json'}")

    if metrics["passed"]:
        def predict(live_features: pd.DataFrame) -> pd.DataFrame:
            # Must mirror scripts/numerai_data.py's preprocessing exactly: scale
            # raw [0, 4] features to [0, 1], then standardize with the fit-time
            # mean/std baked in at export time.
            x = live_features[feature_set].to_numpy(dtype=np.float32) / 4.0
            x = (x - mean) / std
            preds = model.predict(x)
            return pd.Series(preds, index=live_features.index).to_frame("prediction")

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
```

The rest of the file (plotting, `if __name__ == "__main__":`) is unchanged.

- [ ] **Step 3: Run it for real**

This repo has no unit test suite (data-loading and training are only exercised by actually
running the pipelines against real data — see how `train.mojo`/`gpu_kernels.mojo` verify
themselves). Verify this change the same way: run the whole script end to end.

Run (plain system Python — pixi is only needed for Mojo in this repo):

```bash
python src/xgboost-model/main.py
```

Expected: training runs as before (takes a few minutes — 2000 estimators over the fit split),
then prints the metrics dict, then either:
- `wrote <path>\artifacts\xgboost_model.pkl (NN.N KB)`, if `metrics["passed"]` is true, or
- the `holdout check failed ...` message, if it isn't.

Either way, `artifacts/xgboost_model.json` is written and the two plot windows still appear, same
as before this change.

- [ ] **Step 4: Verify the pickle round-trips and doesn't touch the Mojo model's artifact**

If Step 3 produced `artifacts/xgboost_model.pkl` (holdout passed), confirm it's a loadable,
callable predictor that returns the right shape, and that it did not touch
`artifacts/model.pkl` (the Mojo pipeline's output, if present from an earlier `pixi run train` /
`pixi run submit-package`):

```bash
python -c "
import cloudpickle
import pandas as pd
from pathlib import Path

ARTIFACTS_DIR = Path('artifacts')
with open(ARTIFACTS_DIR / 'xgboost_model.pkl', 'rb') as f:
    predict = cloudpickle.load(f)

df = pd.read_parquet(Path('data/v5.3/train.parquet')).head(5)
import json
feature_set = json.load(open(Path('data/features.json')))['feature_sets']['small']
out = predict(df[feature_set])
print(out)
assert list(out.columns) == ['prediction']
assert len(out) == 5
print('OK: xgboost_model.pkl round-trips')

model_pkl = ARTIFACTS_DIR / 'model.pkl'
if model_pkl.exists():
    print(f'model.pkl still present, untouched by this run: {model_pkl.stat()}')
"
```

Expected: prints a 5-row, one-column (`prediction`) DataFrame, then `OK: xgboost_model.pkl
round-trips`, and (if `artifacts/model.pkl` existed already) confirms it's still there.

If Step 3 instead printed the `holdout check failed` message, skip this step — there's no pickle
to check — and instead confirm `artifacts/xgboost_model.pkl` was **not** written:

```bash
python -c "from pathlib import Path; assert not Path('artifacts/xgboost_model.pkl').exists(); print('OK: no pickle written on failed holdout check')"
```

- [ ] **Step 5: Commit**

```bash
git add src/xgboost-model/main.py
git commit -m "$(cat <<'EOF'
feat: save XGBoost model as a Numerai-submittable pickle

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
