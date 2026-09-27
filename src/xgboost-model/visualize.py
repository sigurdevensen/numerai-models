from matplotlib import pyplot as plt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate import per_era_corr_series

def plot_corr(predictions, holdout_y, holdout_eras):
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