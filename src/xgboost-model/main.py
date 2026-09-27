import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_VERSION = "v5.3"

def load_data(path: str, feature_set: list) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        path (str): The path to the CSV file.
    """
    train = pd.read_parquet(
        path,
        columns=["era", "target", "target_ender_20"] + feature_set
    )

    return train

def load_feature_set(feature_set_name="small"):
    with open(DATA_DIR / "features.json") as f:
        meta = json.load(f)
    return meta["feature_sets"][feature_set_name]

def main():
    # Example usage
    path = DATA_DIR / DATA_VERSION / "train.parquet"
    feature_set = load_feature_set("faith")
    data = load_data(path, feature_set)
    data.groupby("era").size().plot(
            title="Number of rows per era",
            figsize=(5,3),
            xlabel="Era"
        )
    plt.tight_layout()
    #output_path = Path(__file__).resolve().parent / "rows_per_era.png"
    #plt.savefig(output_path)
    #print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()