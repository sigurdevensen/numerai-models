# numerai-models

## Setup

Mojo doesn't run natively on Windows. This repo runs inside WSL2 (Ubuntu) via
[pixi](https://pixi.sh):

```powershell
wsl --install -d Ubuntu   # one-time; needs WSL2's VM platform already enabled
```

Then, inside that Ubuntu distro, from the repo root:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
```

`pixi.toml` pins `mojo`, `python`, `numpy`, `pandas`, `pyarrow`, and `max` (the
GPU/accelerator runtime) from Modular's conda channel, plus `numerapi`,
`cloudpickle`, and `numerai-tools` from PyPI.

## Running it

```bash
pixi run pull-data        # downloads data/v5.3/train.parquet if missing (~3GB)
pixi run train             # mojo run src/train.mojo: fit -> validate -> export
pixi run submit-package    # packages artifacts/ into artifacts/model.pkl
```

`train` refuses to write `artifacts/` at all if the holdout check fails (mean
CORR <= 0 or Sharpe <= 0.3 across the 60 held-out eras -- see
`MIN_MEAN_CORR`/`MIN_SHARPE` in `src/train.mojo`), so a stale `model.pkl` is
never built from a model that didn't validate.

The last training run here: 2,363,699 fit rows / 320,148 holdout rows (era-based
split, 60 holdout eras + 13 embargo eras), 42 features (the "small" feature
set), 300 epochs of full-batch gradient descent (~2 minutes on CPU). Holdout
result: **mean CORR 0.0298, Sharpe 2.06** across 60 eras -- see
`artifacts/model_meta.json` after running `pixi run train` for the current
numbers.

To actually submit to Numerai Compute (this step is **not** run automatically
by anything in this repo -- it's a real, live change to your model):

```bash
export NUMERAI_PUBLIC_ID=...
export NUMERAI_SECRET_KEY=...   # needs the upload_submission scope
pixi run python scripts/submit.py upload --model-id <your-model-uuid>
```

`docker_image` (default `"Python 3.11"`) must match the Python version Numerai
runs the pickle under -- check `NumerAPI().model_upload_docker_images()` for
current options; a mismatch fails at pickle-execution time on Numerai's side,
not at upload time.
