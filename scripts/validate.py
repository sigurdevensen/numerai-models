"""Per-era CORR validation using Numerai's own scoring implementation.

Called from Mojo (src/train.mojo) via Python interop, and usable standalone from Python.
"""
import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr


def per_era_corr(predictions, targets, eras, min_mean=0.0, min_sharpe=0.0):
    """Returns {mean, std, sharpe, n_eras, passed} for holdout predictions, era by era.

    `passed` gates whether src/train.mojo proceeds to export/submit: mean CORR
    must beat `min_mean` and the Sharpe (mean/std across eras) must beat
    `min_sharpe`, i.e. the signal must be both positive and not noise-sized.
    """
    df = pd.DataFrame(
        {
            "era": np.asarray(eras),
            "prediction": np.asarray(predictions, dtype=np.float64),
            "target": np.asarray(targets, dtype=np.float64),
        }
    )
    corrs = df.groupby("era").apply(
        lambda e: numerai_corr(e[["prediction"]], e["target"]).iloc[0],
        include_groups=False,
    )
    mean = float(corrs.mean())
    std = float(corrs.std(ddof=0))
    sharpe = mean / std if std > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "n_eras": int(len(corrs)),
        "passed": bool(mean > min_mean and sharpe > min_sharpe),
    }
