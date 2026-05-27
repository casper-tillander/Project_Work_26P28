from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs_regressor"
META_COLS = ["Subject", "Timestamp", "Activity", "Gait", "Gait_Pct", "Reps"]


def find_file(user_path, candidates, name):
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
        return path
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {name}. Searched:\n{searched}")


def default_candidates(filename):
    return [
        PROJECT_ROOT / "data" / filename,
        PROJECT_ROOT / "Data" / filename,
        PROJECT_ROOT.parent / "data" / filename,
        PROJECT_ROOT.parent / "Data" / filename,
        Path(r"C:\Users\ansku\Desktop\School\PROJECT\testdata") / filename,
    ]


def target_columns(df, requested):
    available = [c for c in df.columns if "Target" in c]
    if not available:
        raise ValueError("No target columns found. Expected columns containing 'Target'.")
    if requested:
        missing = [c for c in requested if c not in available]
        if missing:
            raise ValueError(f"Missing targets: {missing}\nAvailable: {available}")
        return requested
    return available


def feature_columns(df, targets):
    excluded = set(META_COLS + targets)
    cols = [c for c in df.columns if c not in excluded and "Target" not in c]
    if not cols:
        raise ValueError("No feature columns found.")
    return cols


def make_test_vectors(X, y_true, y_pred, max_vectors=5):
    n = min(max_vectors, len(X))
    vectors = []
    for i in range(n):
        vectors.append({
            "features": X[i].astype(float).tolist(),
            "true_values": np.atleast_1d(y_true[i]).astype(float).tolist(),
            "pred_values": np.atleast_1d(y_pred[i]).astype(float).tolist(),
        })
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--target", nargs="*", default=None)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = find_file(args.train_data, default_candidates("09_train_features.pkl"), "train data")
    test_path = find_file(args.test_data, default_candidates("09_test_features.pkl"), "test data")

    print(f"Loading train data: {train_path}")
    print(f"Loading test data:  {test_path}")

    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    targets = target_columns(train_df, args.target)
    features = feature_columns(train_df, targets)

    missing_features = [c for c in features if c not in test_df.columns]
    missing_targets = [c for c in targets if c not in test_df.columns]
    if missing_features or missing_targets:
        raise ValueError(f"Train/test columns do not match. Missing features: {missing_features}, missing targets: {missing_targets}")

    X_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df[targets].to_numpy(dtype=np.float32)
    X_test = test_df[features].to_numpy(dtype=np.float32)
    y_test = test_df[targets].to_numpy(dtype=np.float32)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )

    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print(f"Regressor Train R2: {r2_score(y_train, pred_train):.4f}, RMSE: {np.sqrt(mean_squared_error(y_train, pred_train)):.4f}")
    print(f"Regressor Test R2:  {r2_score(y_test, pred_test):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, pred_test)):.4f}")
    print(f"Features: {len(features)}")
    print(f"Targets:  {len(targets)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "regressor_model.pkl")
    (OUTPUT_DIR / "regressor_feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "regressor_target_names.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "regressor_test_vectors.json").write_text(json.dumps(make_test_vectors(X_test, y_test, pred_test), indent=2), encoding="utf-8")

    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
