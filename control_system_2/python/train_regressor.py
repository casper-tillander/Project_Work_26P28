from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Path configurations
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "outputs_regressor"

# Meta columns
META_COLS = ["Activity", "Gait_Pct"]

def extract_features(raw_df, window_size=410, step_size=102):
    """
    Converts raw continuous data into sliding window features 
    required by the Random Forest model.
    """
    print("Extracting features from raw data...")
    ta_cols = [c for c in raw_df.columns if "EMG_TA" in c]
    gm_cols = [c for c in raw_df.columns if "EMG_GM" in c]
    imu_cols = [c for c in raw_df.columns if any(s in c for s in ["-f", "-c", "-q", "-h"])]
    gonio_cols = [c for c in raw_df.columns if c.startswith("G_")]

    rows = []
    # Slide a window across the raw data
    for start in range(0, len(raw_df) - window_size + 1, step_size):
        end = start + window_size
        window = raw_df.iloc[start:end]

        # Ensure window stays within a single repetition and activity
        if window['Activity'].nunique() > 1 or window['Reps'].nunique() > 1:
            continue

        last_row = window.iloc[-1]
        feat = {
            "Activity": last_row["Activity"],
            "Gait_Pct": last_row["Gait"]
        }
        
        # Calculate features
        feat["EMG_TA_MAV"] = np.mean(np.abs(window[ta_cols].to_numpy()))
        feat["EMG_GM_MAV"] = np.mean(np.abs(window[gm_cols].to_numpy()))
        for col in imu_cols:
            feat[f"IMU_{col}_Mean"] = window[col].mean()

        # Set joint angles as targets
        for col in gonio_cols:
            feat[f"{col}_Target"] = last_row[col]
            
        rows.append(feat)

    return pd.DataFrame(rows)


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
    parser.add_argument("--target", nargs="*", default=None)
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Define paths
    train_path = DATA_DIR / "training_dataset.pkl"
    test_path = DATA_DIR / "testing_dataset.pkl"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Could not find training or testing datasets in {DATA_DIR}. Please run split_database.py first.")

    print(f"Loading raw train data: {train_path}")
    raw_train_df = pd.read_pickle(train_path)
    
    print(f"Loading raw test data:  {test_path}")
    raw_test_df = pd.read_pickle(test_path)

    # Extract features
    train_df = extract_features(raw_train_df)
    test_df = extract_features(raw_test_df)

    # Organize columns
    targets = target_columns(train_df, args.target)
    features = feature_columns(train_df, targets)

    # Verify column match between train and test
    missing_features = [c for c in features if c not in test_df.columns]
    missing_targets = [c for c in targets if c not in test_df.columns]
    if missing_features or missing_targets:
        raise ValueError(f"Train/test columns do not match. Missing features: {missing_features}, missing targets: {missing_targets}")

    # Convert to Numpy arrays
    X_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df[targets].to_numpy(dtype=np.float32)
    X_test = test_df[features].to_numpy(dtype=np.float32)
    y_test = test_df[targets].to_numpy(dtype=np.float32)

    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )

    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    # Evaluate model
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print(f"Regressor Train R2: {r2_score(y_train, pred_train):.4f}, RMSE: {np.sqrt(mean_squared_error(y_train, pred_train)):.4f}")
    print(f"Regressor Test R2:  {r2_score(y_test, pred_test):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, pred_test)):.4f}")
    print(f"Features used: {len(features)}")
    print(f"Targets predicted: {len(targets)}")

    # Save outputs
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "regressor_model.pkl")
    (OUTPUT_DIR / "regressor_feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "regressor_target_names.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "regressor_test_vectors.json").write_text(json.dumps(make_test_vectors(X_test, y_test, pred_test), indent=2), encoding="utf-8")

    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()