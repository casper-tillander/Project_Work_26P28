from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

subjects = ['01', '09', '10']
window_size = 410
step_size = 205

all_extracted_features = []

for sub in subjects:
    print(f"Processing subject {sub}...")
    file_name = DATA_DIR / f"{sub}_filtered_all_data.pkl"

    if not file_name.exists():
        print(f"Skipping subject {sub}: file not found -> {file_name}")
        continue

    df = pd.read_pickle(file_name)

    ta_cols = [f"EMG_TA_CH_{i}" for i in range(64)]
    gm_cols = [f"EMG_GM_CH_{i}" for i in range(64)]
    imu_cols = [col for col in df.columns if '-' in col]
    gonio_cols = [col for col in df.columns if col.startswith('G_')]

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]

        win_feats = {}

        # Metadata
        win_feats['Subject'] = sub
        win_feats['Timestamp'] = window['Timestamp'].iloc[-1]
        win_feats['Activity'] = window['Activity'].iloc[-1]
        win_feats['Gait'] = window['Gait'].iloc[-1]
        win_feats['Reps'] = window['Reps'].iloc[-1]

        # EMG TA features
        ta_data = window[ta_cols].values.astype(float)
        win_feats["EMG_TA_MAV_Mean"] = np.mean(np.mean(np.abs(ta_data), axis=0))
        win_feats["EMG_TA_WL_Mean"] = np.mean(np.sum(np.abs(np.diff(ta_data, axis=0)), axis=0))
        win_feats["EMG_TA_ZC_Mean"] = np.mean(np.sum((ta_data[:-1] * ta_data[1:]) < 0, axis=0))

        # EMG GM features
        gm_data = window[gm_cols].values.astype(float)
        win_feats["EMG_GM_MAV_Mean"] = np.mean(np.mean(np.abs(gm_data), axis=0))
        win_feats["EMG_GM_WL_Mean"] = np.mean(np.sum(np.abs(np.diff(gm_data, axis=0)), axis=0))
        win_feats["EMG_GM_ZC_Mean"] = np.mean(np.sum((gm_data[:-1] * gm_data[1:]) < 0, axis=0))

        # IMU features
        imu_data = window[imu_cols].values.astype(float)
        imu_mean = np.mean(imu_data, axis=0)
        imu_std = np.std(imu_data, axis=0)
        imu_jerk = np.std(np.diff(imu_data, axis=0), axis=0)

        for i, col in enumerate(imu_cols):
            win_feats[f"IMU_{col}_Mean"] = imu_mean[i]
            win_feats[f"IMU_{col}_Std"] = imu_std[i]
            win_feats[f"IMU_{col}_Jerk"] = imu_jerk[i]

        # Goniometer targets
        gonio_data = window[gonio_cols].values.astype(float)
        gonio_mean = np.mean(gonio_data, axis=0)

        for i, col in enumerate(gonio_cols):
            win_feats[f"{col}_Target"] = gonio_mean[i]

        all_extracted_features.append(win_feats)

if not all_extracted_features:
    raise ValueError("No features were extracted. Check data files and column names.")

feature_df = pd.DataFrame(all_extracted_features)

feature_pkl_path = DATA_DIR / "feature_database.pkl"
feature_csv_path = DATA_DIR / "feature_database.csv"

feature_df.to_pickle(feature_pkl_path)
feature_df.to_csv(feature_csv_path, index=False)

print(f"Saved to {feature_pkl_path}")
print(f"Saved to {feature_csv_path}")

print("Final feature database shape:", feature_df.shape)
print("Subjects included:", feature_df["Subject"].unique())
print("Activities included:", feature_df["Activity"].unique())