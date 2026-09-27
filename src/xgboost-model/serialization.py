import cloudpickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

def generate_pkl(predict):
    pkl_path = ARTIFACTS_DIR / "xgboost_model.pkl"
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(predict, f)
    print(f"wrote {pkl_path.resolve()} ({pkl_path.stat().st_size / 1e3:.1f} KB)")

def generate_model_json(model):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model.save_model(ARTIFACTS_DIR / "xgboost_model.json")
    print(f"saved model to {ARTIFACTS_DIR / 'xgboost_model.json'}")