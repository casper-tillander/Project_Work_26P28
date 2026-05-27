import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix


# Loaad data and other necessary constants

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "feature_database.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_COL = "Activity"
RANDOM_STATE = 42

RF_PARAM_GRID = {
    "n_estimators": [10, 20],      # small for easy C export
    "max_depth": [3, 5],
    "min_samples_leaf": [5, 10]
}


# filter data

def load_and_filter_data(path):
    df = pd.read_pickle(path)
    
    df = df[~df[TARGET_COL].astype(str).str.contains('_')]

    print(f"Loaded data: {df.shape}")
    print(f"Unique subjects: {df['Subject'].nunique()}")
    print(f"Classes: {df[TARGET_COL].unique()}")

    return df


# select target columns

def get_feature_columns(df):
    meta_cols = ['Subject', 'Timestamp', 'Activity', 'Gait', 'Reps']
    gonio_cols = [col for col in df.columns if "_Target" in col]

    feature_cols = [col for col in df.columns if col not in meta_cols + gonio_cols]

    print(f"Number of features: {len(feature_cols)}")

    return feature_cols


# eccode labels

def encode_labels(df):
    classes = sorted(df[TARGET_COL].unique())
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    inverse_map = {v: k for k, v in label_map.items()}

    y = df[TARGET_COL].map(label_map).values

    print("Label mapping:", label_map)

    return y, label_map, inverse_map


# execution of grid search

def run_grid_search(X, y, groups):
    logo = LeaveOneGroupOut()

    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=RF_PARAM_GRID,
        cv=logo,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y, groups=groups)

    print("\nBest parameters:", grid.best_params_)
    print("Best accuracy:", grid.best_score_)

    return grid


# train model with best params

def train_final_model(X, y, best_params):
    model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE
    )

    model.fit(X, y)

    return model


# save model, feature names, labels

def save_artifacts(model, feature_names, label_map):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=4)

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)

    print("Artifacts saved.")

# export for C implementation

def export_test_vectors(X, y, model, inverse_map, num_samples=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    indices = np.random.choice(len(X), num_samples, replace=False)

    rows = []
    for idx in indices:
        features = X[idx]
        true_label = y[idx]
        pred_label = model.predict([features])[0]

        rows.append({
            "features": features.tolist(),
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "true_name": inverse_map[true_label],
            "pred_name": inverse_map[pred_label]
        })

    with open(os.path.join(OUTPUT_DIR, "test_vectors.json"), "w") as f:
        json.dump(rows, f, indent=4)

    print("Test vectors exported.")




def main():
    df = load_and_filter_data(DATA_PATH)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    groups = df["Subject"].values
    y, label_map, inverse_map = encode_labels(df)
    grid = run_grid_search(X, y, groups)

    best_params = grid.best_params_
    model = train_final_model(X, y, best_params)
    save_artifacts(model, feature_cols, label_map)
    export_test_vectors(X, y, model, inverse_map)

    print("Training completed.")


if __name__ == "__main__":
    main()