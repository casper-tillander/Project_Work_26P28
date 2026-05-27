from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SUBJECT = "09"          # subject id
TRAIN_RATIO = 0.8       # train split
GAP_SAMPLES = 410       # gap size
WINDOW_SIZE = 410       # window size
STEP_SIZE = 205         # step size


def find_columns(df: pd.DataFrame):
    ta_cols = [f"EMG_TA_CH_{i}" for i in range(64) if f"EMG_TA_CH_{i}" in df.columns]
    gm_cols = [f"EMG_GM_CH_{i}" for i in range(64) if f"EMG_GM_CH_{i}" in df.columns]
    imu_cols = [col for col in df.columns if "-" in col]
    gonio_cols = [col for col in df.columns if col.startswith("G_")]
    return ta_cols, gm_cols, imu_cols, gonio_cols


def validate_columns(df: pd.DataFrame):
    ta_cols, gm_cols, imu_cols, gonio_cols = find_columns(df)

    if len(ta_cols) != 64:
        raise ValueError(f"Expected 64 TA channels, found {len(ta_cols)}")
    if len(gm_cols) != 64:
        raise ValueError(f"Expected 64 GM channels, found {len(gm_cols)}")
    if not imu_cols:
        raise ValueError("No IMU columns found.")
    if not gonio_cols:
        raise ValueError("No goniometer columns found.")

    return ta_cols, gm_cols, imu_cols, gonio_cols


def split_raw_df_by_time(df: pd.DataFrame, train_ratio: float, gap_samples: int):
    n = len(df)

    if n < WINDOW_SIZE * 3:
        raise ValueError("Dataset too short.")

    train_end = int(n * train_ratio)      # train end
    test_start = train_end + gap_samples  # test start

    if train_end < WINDOW_SIZE:
        raise ValueError("Train too short.")
    if n - test_start < WINDOW_SIZE:
        raise ValueError("Test too short.")

    train_raw = df.iloc[:train_end].copy()      # train part
    test_raw = df.iloc[test_start:].copy()      # test part

    return train_raw, test_raw, train_end, test_start


def extract_features(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    ta_cols, gm_cols, imu_cols, gonio_cols = validate_columns(df)
    rows = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]   # window slice
        feat = {}

        # metadata
        feat["Subject"] = subject_id
        feat["Window_Start"] = start
        feat["Window_End"] = end
        feat["Timestamp"] = window["Timestamp"].iloc[-1] if "Timestamp" in window.columns else np.nan
        feat["Activity"] = (
            window["Activity"].mode().iloc[0]
            if "Activity" in window.columns and not window["Activity"].dropna().empty
            else np.nan
        )
        feat["Gait"] = (
            window["Gait"].mode().iloc[0]
            if "Gait" in window.columns and not window["Gait"].dropna().empty
            else np.nan
        )
        feat["Reps"] = (
            window["Reps"].mode().iloc[0]
            if "Reps" in window.columns and not window["Reps"].dropna().empty
            else np.nan
        )

        # TA features
        ta_data = window[ta_cols].to_numpy(dtype=float)
        feat["EMG_TA_MAV_Mean"] = np.mean(np.mean(np.abs(ta_data), axis=0))
        feat["EMG_TA_WL_Mean"] = np.mean(np.sum(np.abs(np.diff(ta_data, axis=0)), axis=0))
        feat["EMG_TA_ZC_Mean"] = np.mean(np.sum((ta_data[:-1] * ta_data[1:]) < 0, axis=0))

        # GM features
        gm_data = window[gm_cols].to_numpy(dtype=float)
        feat["EMG_GM_MAV_Mean"] = np.mean(np.mean(np.abs(gm_data), axis=0))
        feat["EMG_GM_WL_Mean"] = np.mean(np.sum(np.abs(np.diff(gm_data, axis=0)), axis=0))
        feat["EMG_GM_ZC_Mean"] = np.mean(np.sum((gm_data[:-1] * gm_data[1:]) < 0, axis=0))

        # IMU features
        imu_data = window[imu_cols].to_numpy(dtype=float)
        imu_mean = np.mean(imu_data, axis=0)
        imu_std = np.std(imu_data, axis=0)
        imu_jerk = np.std(np.diff(imu_data, axis=0), axis=0)

        for i, col in enumerate(imu_cols):
            feat[f"IMU_{col}_Mean"] = imu_mean[i]
            feat[f"IMU_{col}_Std"] = imu_std[i]
            feat[f"IMU_{col}_Jerk"] = imu_jerk[i]

        # targets
        gonio_data = window[gonio_cols].to_numpy(dtype=float)
        gonio_mean = np.mean(gonio_data, axis=0)

        for i, col in enumerate(gonio_cols):
            feat[f"{col}_Target"] = gonio_mean[i]

        rows.append(feat)   # append row

    if not rows:
        raise ValueError("No features extracted.")

    return pd.DataFrame(rows)


def main():
    subject_id = SUBJECT
    input_path = DATA_DIR / f"{subject_id}_filtered_all_data.pkl"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading {input_path} ...")
    df = pd.read_pickle(input_path)   # load data
    print(f"Raw dataframe shape: {df.shape}")

    train_raw, test_raw, train_end, test_start = split_raw_df_by_time(
        df,
        train_ratio=TRAIN_RATIO,
        gap_samples=GAP_SAMPLES,
    )

    print(f"Train raw rows: {train_raw.shape[0]}")
    print(f"Gap samples: {GAP_SAMPLES}")
    print(f"Test raw rows: {test_raw.shape[0]}")

    train_features = extract_features(train_raw, subject_id)   # train features
    test_features = extract_features(test_raw, subject_id)     # test features

    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    train_pkl = DATA_DIR / f"{subject_id}_train_features.pkl"
    test_pkl = DATA_DIR / f"{subject_id}_test_features.pkl"

    train_features.to_pickle(train_pkl)   # save train
    test_features.to_pickle(test_pkl)     # save test

    split_meta = {
        "subject": subject_id,
        "rows_total": int(len(df)),
        "train_ratio": TRAIN_RATIO,
        "gap_samples": GAP_SAMPLES,
        "train_end": int(train_end),
        "test_start": int(test_start),
        "train_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
    }

    with open(DATA_DIR / f"{subject_id}_split_info.json", "w") as f:
        json.dump(split_meta, f, indent=2)   # save meta

    print("Done.")


if __name__ == "__main__":
    main()