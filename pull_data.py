import json
from numerapi import NumerAPI

napi = NumerAPI()



def download_features(DATA_VERSION):
    """Download the features.json file from Numerai.""" 
    napi.download_dataset(f"{DATA_VERSION}/features.json", f"data//features.json")

def download_training_data(DATA_VERSION):
    """Download the training data from Numerai."""
    napi.download_dataset(f"{DATA_VERSION}/train.parquet", f"data//train.parquet")

if __name__ == "__main__":
    DATA_VERSION = "v5.3"
    download_features(DATA_VERSION)
    download_training_data(DATA_VERSION)