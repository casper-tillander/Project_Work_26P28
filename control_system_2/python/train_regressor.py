from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Path configurations
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "outputs_regressor"

# Meta columns
META_COLS = ["Activity", "Gait_Pct"]

# Sampling parameters
SAMPLE_RATE_HZ = 2048
CHUNK_SIZE_MS = 150
CHUNK_SAMPLES = int((SAMPLE_RATE_HZ * CHUNK_SIZE_MS) / 1000) # Equals 307
STEP_SAMPLES = CHUNK_SAMPLES // 2  # 50% overlap, equals 153

def extract_features(raw_df, window_size=CHUNK_SAMPLES, step_size=STEP_SAMPLES):
    """
    Converts raw continuous data into sliding window features 
    required by the Random Forest model.
    """
    print(f"Extracting features using window_size={window_size} (150ms) and step_size={step_size}...")
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
        
        # Calculate features (Mean Absolute Value for EMG)
        # np.mean matches the C code's logic: sum / (CHUNK_SAMPLES * NUM_COLS)
        feat["EMG_TA_MAV"] = np.mean(np.abs(window[ta_cols].to_numpy()))
        feat["EMG_GM_MAV"] = np.mean(np.abs(window[gm_cols].to_numpy()))
        
        # Calculate IMU Means
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
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
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

    # Overall metrics
    print("\n" + "="*65)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*65)
    
    # Train set metrics
    train_r2 = r2_score(y_train, pred_train)
    train_mse = mean_squared_error(y_train, pred_train)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, pred_train)
    print(f"[OVERALL - TRAIN] R2: {train_r2:.4f} | RMSE: {train_rmse:.4f} | MSE: {train_mse:.4f} | MAE: {train_mae:.4f}")

    # Test set metrics
    test_r2 = r2_score(y_test, pred_test)
    test_mse = mean_squared_error(y_test, pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, pred_test)
    print(f"[OVERALL - TEST]  R2: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")
    
    print("-" * 65)
    print("PERFORMANCE BY ACTIVITY (TEST SET)")
    print("-" * 65)
    
    # Calculate metrics grouped by activity
    activities = test_df['Activity'].unique()
    for act in activities:
        # Create a mask for the current activity
        mask = test_df['Activity'] == act
        
        y_test_act = y_test[mask]
        pred_test_act = pred_test[mask]
        
        act_r2 = r2_score(y_test_act, pred_test_act)
        act_mse = mean_squared_error(y_test_act, pred_test_act)
        act_rmse = np.sqrt(act_mse)
        act_mae = mean_absolute_error(y_test_act, pred_test_act)
        
        print(f"{act.capitalize():<15} -> R2: {act_r2:>6.4f} | RMSE: {act_rmse:>6.4f} | MSE: {act_mse:>6.4f} | MAE: {act_mae:>6.4f}")
        
    print("="*65 + "\n")

    print(f"Features used: {len(features)}")
    print(f"Targets predicted: {len(targets)}")

    # Save outputs
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "regressor_model.pkl")
    (OUTPUT_DIR / "regressor_feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "regressor_target_names.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "regressor_test_vectors.json").write_text(json.dumps(make_test_vectors(X_test, y_test, pred_test), indent=2), encoding="utf-8")

    print(f"Saved outputs to: {OUTPUT_DIR}")

    # Plotting
    print("Generating plots...")
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.labelpad": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,       
        "savefig.dpi": 300,      
        "savefig.bbox": "tight"  
    })
    
    sns.set_style("ticks")

    dominant_target_name = targets[0]
    # Dynamically extract the exact raw column name (e.g., "G_Ankle_Target" -> "G_Ankle")
    raw_target_col = dominant_target_name.replace("_Target", "")
    
    # Assign continuous time to the raw 2048 Hz data per activity
    raw_step_s = 1.0 / SAMPLE_RATE_HZ
    raw_test_df['Time (s)'] = raw_test_df.groupby('Activity').cumcount() * raw_step_s
    
    # Extract timestamps for predictions to perfectly align them with raw data gaps
    pred_times = []
    for start in range(0, len(raw_test_df) - CHUNK_SAMPLES + 1, STEP_SAMPLES):
        end = start + CHUNK_SAMPLES
        window = raw_test_df.iloc[start:end]
        if window['Activity'].nunique() > 1 or window['Reps'].nunique() > 1:
            continue
        # The prediction's ground truth time is the very end of the window
        pred_times.append(raw_test_df['Time (s)'].iloc[end - 1])

    # Prepare predictions dataframe (13.3 Hz)
    y_pred_dom = pred_test[:, 0] if pred_test.ndim > 1 else pred_test
    plot_df = pd.DataFrame({
        'Predicted Angle': y_pred_dom,
        'Activity': test_df['Activity'].values,
        'Time (s)': pred_times
    })
    
    plot_df['Activity'] = plot_df['Activity'].str.capitalize()
    
    # Prepare raw dataFrame (2048 Hz)
    raw_plot_df = raw_test_df[['Activity', 'Time (s)', raw_target_col]].copy()
    raw_plot_df['Activity'] = raw_plot_df['Activity'].str.capitalize()

    # Limit the first two activities to 5 seconds max
    first_two_acts = plot_df['Activity'].unique()[:2]
    
    pred_mask = ~(plot_df['Activity'].isin(first_two_acts) & (plot_df['Time (s)'] > 5.0))
    plot_df = plot_df[pred_mask]
    
    raw_mask = ~(raw_plot_df['Activity'].isin(first_two_acts) & (raw_plot_df['Time (s)'] > 5.0))
    raw_plot_df = raw_plot_df[raw_mask]

    # Build the subplots natively to handle differing lengths
    plot_activities = plot_df['Activity'].unique()
    fig, axes = plt.subplots(nrows=len(plot_activities), ncols=1, 
                             figsize=(10, 2.5 * len(plot_activities)), 
                             sharex=False, sharey=False)
    
    if len(plot_activities) == 1:
        axes = [axes]
        
    palette = {"True Angle": "#2C3E50", "Predicted Angle": "#C0392B"} 
    
    for ax, act in zip(axes, plot_activities):
        act_raw = raw_plot_df[raw_plot_df['Activity'] == act]
        act_pred = plot_df[plot_df['Activity'] == act]
        
        # Smooth true line (2048 Hz)
        ax.plot(act_raw['Time (s)'], act_raw[raw_target_col], 
                color=palette["True Angle"], label="True Angle", 
                linewidth=1.5, alpha=0.9)
        
        # Overlay stepped predictions (13.3 Hz)
        ax.plot(act_pred['Time (s)'], act_pred['Predicted Angle'], 
                color=palette["Predicted Angle"], label="Predicted Angle",
                linestyle='--', marker='.', markersize=4, linewidth=1.5, alpha=0.9)
                
        ax.set_title(act, size=12, weight='bold')
        
        sns.despine(ax=ax, trim=False, offset=5)
        
    axes[-1].set_xlabel("Time (s)")

    # Global labels and layout adjustment
    fig.supylabel("Dominant Ankle Angle (deg)", fontsize=12)
    fig.suptitle("True vs Predicted Dominant Ankle Angle Across Activities", fontsize=14, weight='bold')
    
    # Legend
    axes[0].legend(title="", frameon=False, loc="center right", bbox_to_anchor=(1.18, 0.5))
    fig.subplots_adjust(top=0.90, left=0.10, right=0.88, hspace=0.4) 

    plot_path = OUTPUT_DIR / "time_series_by_activity.png"
    plt.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
