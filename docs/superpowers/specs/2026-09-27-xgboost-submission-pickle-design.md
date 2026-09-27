# XGBoost submission pickle

## Problem

`src/xgboost-model/main.py` trains an XGBoost baseline and saves it with
`model.save_model(artifacts/xgboost_model.json)` — XGBoost's native format. Numerai Compute
requires a pickled Python `predict(live_features) -> DataFrame` callable (the same contract
`scripts/submit.py::package` already produces for the Mojo ridge model); the native JSON format
can't be submitted as-is. There is currently no way to get the XGBoost model into a submittable
format.

## Goals

- After training, produce a cloudpickled `predict()` callable for the XGBoost model that Numerai
  Compute can run directly, following the same closure contract `scripts/submit.py` uses.
- Don't collide with the Mojo pipeline's `artifacts/model.pkl`.
- Don't submit a model that hasn't demonstrated real (non-noise) holdout signal.

## Non-goals

- No upload step (`scripts/submit.py upload` already exists and is generic enough to point at
  any `pkl_path`/`model_id`; wiring it up for XGBoost is left for later if wanted).
- No changes to `scripts/submit.py` or the Mojo pipeline.
- No intermediate artifact files (`.npy`/`.json`) for mean/std — unlike the Mojo pipeline, this
  script never crosses a Mojo/Python boundary, so it can build and cloudpickle the closure
  directly in the same process instead of exporting raw arrays and reloading them.

## Design

In `src/xgboost-model/main.py`, after computing holdout predictions:

1. Score holdout with a validation bar matching the Mojo pipeline's:
   `per_era_corr(predictions, holdout_y, holdout_eras, min_mean=0.0, min_sharpe=0.3)`.
2. If `metrics["passed"]`:
   - Build `predict(live_features: pd.DataFrame) -> pd.DataFrame`, a closure over the trained
     `model`, `mean`, `std`, and `feature_set`, that reproduces `numerai_data.py`'s preprocessing
     exactly: `x = live_features[feature_set].to_numpy(dtype=np.float32) / 4.0`, then
     `x = (x - mean) / std`, then `model.predict(x)`, returned as a `DataFrame` with column
     `"prediction"` (matching the contract `scripts/submit.py`'s closure already uses).
   - `cloudpickle.dump(predict, ...)` to `artifacts/xgboost_model.pkl`.
   - Print the path and file size (mirroring `submit.py::package`'s log line).
3. If `metrics["passed"]` is false:
   - Print a warning naming the actual mean/sharpe vs. the `0.0`/`0.3` bar, and skip writing the
     pkl.
   - Still save the native `xgboost_model.json` and still show the plots — this script is an
     interactive baseline/exploration tool, not a gated pipeline step, so a failed bar shouldn't
     hard-stop it the way `train.mojo` hard-stops on failure.

No other files change.

## Testing

Run `pixi run python src/xgboost-model/main.py` and confirm:
- `artifacts/xgboost_model.pkl` is written when holdout passes, and is loadable
  (`cloudpickle.load` + calling `predict()` on a small `DataFrame` slice of the feature columns
  returns a one-column `"prediction"` frame).
- `artifacts/model.pkl` (the ridge model's output, if present) is untouched by this run.
- If the bar isn't met, no `xgboost_model.pkl` is written, but `xgboost_model.json` and the plots
  still appear, and a warning is printed with the actual numbers.
