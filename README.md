# numerai-models

Models for [Numerai](https://numer.ai). Two independent projects:

- **`src/claude-model/`** - ridge regression trained from scratch with a native Mojo training loop (CPU + GPU kernels). Made using claude agents.
- **`src/xgboost-model/`** - a plain Python/XGBoost baseline. Made by mostly me.

Both share the same data-loading and CORR scoring code in `scripts/`.

## Structure

```bash
scripts/            shared data loading, scoring, packaging/submission
src/claude-model/    Mojo ridge regression (see its own setup.md)
src/xgboost-model/   XGBoost baseline (main.py)
data/                pulled datasets (gitignored)
artifacts/           trained model outputs (gitignored)
```

## Running it

For mojo model, see [`src/claude-model/setup.md`](src/claude-model/setup.md) for setup and run instructions (runs inside WSL2, since Mojo doesn't run natively on Windows).

**XGBoost baseline**:

```bash
python src/xgboost-model/main.py
```

**Submit to Numerai Compute** (real, live — not automated):

```bash
export NUMERAI_PUBLIC_ID=...
export NUMERAI_SECRET_KEY=...
python scripts/submit.py upload --model-id <your-model-uuid>
```

## Current status

Latest validation diagnostics for the xgboost model (run 2026-09-27):

| Metric | Value |
| --- | --- |
| Validation CORR (mean) | 0.0421 |
| Validation CORR (std) | 0.0530 |
| Validation Sharpe | 0.795 |
| Feature-neutral CORR v3 (mean) | 0.0363 |
| Corr to example predictions | 0.649 |