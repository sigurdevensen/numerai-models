# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Models for the Numerai tournament. Two independent, unconnected model tracks live side by side:

- `src/claude-model/` — a from-scratch ridge-regression model, trained by a native Mojo
  training loop (the point being to actually use Mojo/GPU, not just Python).
- `src/xgboost-model/` — a plain Python/XGBoost baseline script.

Both tracks share the same data-loading and scoring code in `scripts/` via Python (Mojo calls
into it through Python interop; the XGBoost script imports it directly).

## Environment

Mojo does not run natively on Windows. Everything here runs inside WSL2 (Ubuntu) via
[pixi](https://pixi.sh) — see `src/claude-model/setup.md` for one-time setup
(`wsl --install -d Ubuntu`, then `curl -fsSL https://pixi.sh/install.sh | sh` and `pixi install`
inside that distro). `pixi.toml` pins `mojo`, `python` 3.14, `numpy`, `pandas`, `pyarrow`, and
`max` (the GPU/accelerator runtime) from Modular's conda channel, plus `numerapi`,
`cloudpickle`, `numerai-tools`, and `xgboost` from PyPI.
Only use pixi for mojo. Python is ran native on windows, so you use the systems python to run python scripts.

## Commands

```bash
pixi install                          # set up the environment
pixi run pull-data                    # scripts/pull_data.py: downloads data/v5.3/{train.parquet,features.json} (~3GB) if missing
pixi run train                        # fit -> validate -> export (see gotcha below)
pixi run submit-package               # scripts/submit.py package: packages artifacts/ into artifacts/model.pkl
```

**Gotcha:** `pixi.toml`'s `train`, `gpu-check-nvidia`, and `gpu-check-amd` tasks point at
`src/train.mojo` / `src/gpu_kernels.mojo`, but those files actually live under
`src/claude-model/` (moved there per commit `07173b7` without updating `pixi.toml`). Run the
real paths directly until `pixi.toml` is fixed, e.g.:

```bash
pixi run mojo run src/claude-model/train.mojo
pixi run mojo build --target-accelerator sm_80 -o /tmp/gpu_kernels src/claude-model/gpu_kernels.mojo   # NVIDIA
pixi run mojo build --target-accelerator gfx942 -o /tmp/gpu_kernels src/claude-model/gpu_kernels.mojo  # AMD
```

The XGBoost baseline has no pixi task; run it directly: `pixi run python src/xgboost-model/main.py`.

To actually submit to Numerai Compute (a real, live change to your model — not run
automatically by anything here):

```bash
export NUMERAI_PUBLIC_ID=...
export NUMERAI_SECRET_KEY=...   # needs the upload_submission scope
pixi run python scripts/submit.py upload --model-id <your-model-uuid>
```

`--docker-image` (default `"Python 3.11"`) must match the Python version Numerai runs the
pickle under (check `NumerAPI().model_upload_docker_images()`); a mismatch fails at
pickle-execution time on Numerai's side, not at upload time.

## Architecture: the Mojo pipeline (`src/claude-model/`)

`train.mojo` orchestrates the pipeline but delegates anything that needs pandas/pyarrow or
Numerai's reference scoring to Python via Mojo's Python interop — it never reimplements data
loading or CORR scoring in Mojo:

1. `scripts/numerai_data.py::load_split` — reads `data/v5.3/train.parquet`, era-splits it into
   fit/holdout (holding out the most recent `HOLDOUT_ERAS` eras, embargoing `EMBARGO_ERAS`
   immediately before them since the target looks ~13 weekly eras into the future), scales raw
   `[0,4]` features to `[0,1]`, then standardizes using mean/std computed from the fit split
   only. Returns the mean/std alongside the arrays so export/inference can apply the identical
   transform later.
2. `src/claude-model/linear_model.mojo::RidgeRegression` — native Mojo full-batch gradient
   descent (CPU). This is the actual training loop; everything else is glue.
3. `scripts/validate.py::per_era_corr` — scores holdout predictions with Numerai's own
   `numerai_tools.scoring.numerai_corr`, per era, returning mean/std/sharpe.
4. `scripts/export_artifacts.py::save` — only called if the holdout check passes.

**The holdout gate is load-bearing**: `train.mojo` raises and refuses to write `artifacts/` at
all if mean CORR or Sharpe falls at/below `MIN_MEAN_CORR`/`MIN_SHARPE` (constants at the top of
`train.mojo`), so a stale `model.pkl` can never be produced by a run that didn't validate.

**CPU vs GPU split**: `gpu_kernels.mojo` reimplements the same matvec/gradient math as real Mojo
GPU kernels, but lives in a separate file/binary from `train.mojo` rather than as a
runtime-selected branch. This is deliberate, not incidental: Mojo instantiates every reachable
function at compile time regardless of which branch runs, so a `DeviceContext()` call anywhere
in the binary fails to build on a host without a GPU — even behind a `has_accelerator()` check.
Without physical GPU hardware, `gpu_kernels.mojo` can still be compile-checked (not executed) by
cross-compiling with `--target-accelerator` (see Commands above).

`FEATURE_SET` (currently `"small"`, defined against `data/features.json`'s `feature_sets`),
`HOLDOUT_ERAS`, `EMBARGO_ERAS`, `LEARNING_RATE`, `L2`, and `EPOCHS` are all `comptime` constants
at the top of `train.mojo`.

## Submission packaging

Numerai Compute runs a pickled Python `predict(live_features) -> DataFrame` callable — it
cannot run a Mojo binary directly. `scripts/submit.py::package` wraps the weights/mean/std/bias
that `train.mojo` exported to `artifacts/` in a plain numpy `predict()` closure that reproduces
the trained linear model exactly (must mirror `numerai_data.py`'s preprocessing step for step:
scale by `/4.0`, then standardize with the baked-in fit-time mean/std), then cloudpickles it to
`artifacts/model.pkl`.

## Data & artifacts

- `data/v5.3/train.parquet`, `data/features.json` — pulled via `pixi run pull-data`; gitignored.
- `artifacts/` — `weights.npy`, `feature_mean.npy`, `feature_std.npy`, `model_meta.json`
  (bias, feature set, holdout metrics), `model.pkl`. All regenerated by `pixi run train` +
  `pixi run submit-package`; gitignored except `.gitkeep`.
- `example-scripts/` — a vendored clone of Numerai's own example-scripts repo (has its own
  `.git`, gitignored from this repo entirely). Reference notebooks only, not part of this
  project's build.
