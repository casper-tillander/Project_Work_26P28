import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "outputs" / "rf_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    
print("Model loaded")
print("\nNum of trees:", len(model.estimators_))
print("Max depth:", model.max_depth)
print("Num of features:", model.n_features_in_)