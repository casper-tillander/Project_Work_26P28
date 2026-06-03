from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Path configurations
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "outputs_classifier"

# Sampling parameters
SAMPLE_RATE_HZ = 2048
CHUNK_SIZE_MS = 150
CHUNK_SAMPLES = int((SAMPLE_RATE_HZ * CHUNK_SIZE_MS) / 1000) # Equals 307
STEP_SAMPLES = CHUNK_SAMPLES // 2  # 50% overlap, equals 153

def extract_features(raw_df, window_size=CHUNK_SAMPLES, step_size=STEP_SAMPLES):
    """
    Converts raw continuous data into sliding window features 
    required by the classification models.
    """
    print(f"Extracting features using window_size={window_size} (150ms) and step_size={step_size}...")
    ta_cols = [c for c in raw_df.columns if "EMG_TA" in c]
    gm_cols = [c for c in raw_df.columns if "EMG_GM" in c]
    imu_cols = [c for c in raw_df.columns if any(s in c for s in ["-f", "-c", "-q", "-h"])]

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
            "Activity": last_row["Activity"]
        }
        
        # Calculate features (Mean Absolute Value for EMG)
        feat["EMG_TA_MAV"] = np.mean(np.abs(window[ta_cols].to_numpy()))
        feat["EMG_GM_MAV"] = np.mean(np.abs(window[gm_cols].to_numpy()))
        
        # Calculate IMU Means
        for col in imu_cols:
            feat[f"IMU_{col}_Mean"] = window[col].mean()
            
        rows.append(feat)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="", help="Suffix for input/output files (e.g., '_all')")
    args = parser.parse_args()

    # Define paths
    train_path = DATA_DIR / f"training_dataset{args.suffix}.pkl"
    test_path = DATA_DIR / f"testing_dataset{args.suffix}.pkl"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Could not find training or testing datasets: {train_path} or {test_path}")

    print(f"Loading raw train data: {train_path}")
    raw_train_df = pd.read_pickle(train_path)
    
    print(f"Loading raw test data:  {test_path}")
    raw_test_df = pd.read_pickle(test_path)

    # Extract features
    train_df = extract_features(raw_train_df)
    test_df = extract_features(raw_test_df)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Activity"])
    y_test = le.transform(test_df["Activity"])
    class_names = le.classes_

    # Features
    features = [c for c in train_df.columns if c != "Activity"]
    X_train = train_df[features].to_numpy(dtype=np.float32)
    X_test = test_df[features].to_numpy(dtype=np.float32)

    print(f"Features used: {len(features)}")
    print(f"Classes: {class_names}")

    # Define models to compare
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=args.random_state),
        "SVM": SVC(kernel='rbf', probability=True, random_state=args.random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=args.random_state)
    }

    results = {}
    best_model = None
    best_accuracy = -1
    best_model_name = ""

    print("\nComparing models...")
    print(f"{'Model':<20} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 45)
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        
        y_pred_test = model.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        
        results[name] = {"train": acc_train, "test": acc_test}
        print(f"{name:<20} | {acc_train:<10.4f} | {acc_test:<10.4f}")
        
        if acc_test > best_accuracy:
            best_accuracy = acc_test
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with Test Accuracy: {best_accuracy:.4f}")

    # Detailed evaluation for the best model
    y_pred_best = best_model.predict(X_test)
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred_best, target_names=class_names))

    # Save outputs
    experiment_dir = OUTPUT_DIR / f"experiment{args.suffix}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model, experiment_dir / "classifier_model.pkl")
    joblib.dump(le, experiment_dir / "label_encoder.pkl")
    (experiment_dir / "classifier_feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    
    # Save comparison results
    with open(experiment_dir / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.xticks(rotation=45)
    
    weighted_desc = "including weighted" if args.suffix == "_all" else "excluding weighted"
    plt.title(f"Confusion Matrix: {best_model_name}, {weighted_desc}")
    
    cm_path = experiment_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Saved confusion matrix to: {cm_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
