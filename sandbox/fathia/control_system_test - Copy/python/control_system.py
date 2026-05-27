
from pathlib import Path
import argparse
import json
import time
import csv
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "outputs_regressor"

MODEL_PATH         = OUTPUT_DIR / "regressor_model.pkl"
FEATURE_NAMES_PATH = OUTPUT_DIR / "regressor_feature_names.json"
TARGET_NAMES_PATH  = OUTPUT_DIR / "regressor_target_names.json"
LOG_DIR            = OUTPUT_DIR / "control_logs"

SAMPLE_RATE_HZ = 2048
CHUNK_SIZE_MS  = 150
CHUNK_SAMPLES  = (SAMPLE_RATE_HZ * CHUNK_SIZE_MS) // 1000   # 307


# Kp: how hard the exoskeleton pushes when the joint deviates from reference
# Higher = stronger assistance, lower = softer
KP = {
    "G_DA_X_Target":  0.5,   # Dominant ankle sagittal
    "G_DA_Y_Target":  0.2,   # Dominant ankle frontal
    "G_DK_X_Target":  0.8,   # Dominant knee sagittal
    "G_DK_Y_Target":  0.2,   # Dominant knee frontal
    "G_NDA_X_Target": 0.5,   # Non-dominant ankle sagittal
    "G_NDA_Y_Target": 0.2,   # Non-dominant ankle frontal
}

TORQUE_LIMIT = 20.0   # Nm — hard cap per joint


def build_reference_trajectory(train_pkl: Path, target_names: list,
                                activity_filter: str = None,
                                n_bins: int = 101) -> dict:
    """
    Computes mean joint angle at each gait % (0–100) from the training set.
    Returns { joint_name: np.array of shape (101,) }

    This is the reference trajectory — what a healthy person's joint
    should be doing at each point in the gait cycle.
    """
    print(f"Building reference trajectory from: {train_pkl.name}")
    df = pd.read_pickle(train_pkl)

    if activity_filter:
        df = df[df["Activity"] == activity_filter].copy()
        print(f"  Filtered to activity: '{activity_filter}'  ({len(df):,} rows)")
    else:

        df = df[~df["Activity"].str.endswith("_")].copy()
        print(f"  All activities combined  ({len(df):,} rows)")


    gait_col = "Gait" if "Gait" in df.columns else "Gait_Pct"
    if gait_col not in df.columns:
        raise ValueError("Could not find gait percentage column in training data.")

    df["gait_bin"] = np.clip(df[gait_col].round().astype(int), 0, 100)

    reference = {}
    bins = np.arange(n_bins)   # 0..100
    raw_target_map = {t: t.replace("_Target", "") for t in target_names}

    for tgt, raw_col in raw_target_map.items():
        if raw_col not in df.columns:
            print(f"  WARNING: column '{raw_col}' not found, using zeros for {tgt}")
            reference[tgt] = np.zeros(n_bins)
            continue

        means = df.groupby("gait_bin")[raw_col].mean()
        # Fill any missing bins with linear interpolation
        curve = means.reindex(bins).interpolate(method="linear").ffill().bfill().values
        reference[tgt] = curve

    print(f"  Reference trajectory built: {n_bins} gait bins × {len(target_names)} joints")
    return reference

# Feature extraction (same as main.c)


def extract_features(chunk: np.ndarray,
                     ta_idx: list, gm_idx: list, imu_idx: list) -> np.ndarray:
    features = []
    features.append(np.mean(np.abs(chunk[:, ta_idx])))   # EMG TA MAV
    features.append(np.mean(np.abs(chunk[:, gm_idx])))   # EMG GM MAV
    for idx in imu_idx:
        features.append(np.mean(chunk[:, idx]))           # IMU means
    return np.array(features, dtype=np.float32)


def compute_torques(predicted_angles: dict,
                    reference_angles: dict) -> tuple:
    """
    Proportional assistive controller.

    torque = Kp * (reference_angle - predicted_angle)

    Positive torque = assist toward reference.
    Negative torque = resist (joint is overshooting reference).
    Clamped to ±TORQUE_LIMIT.

    Returns (torques dict, errors dict)
    """
    torques = {}
    errors  = {}
    for joint, predicted in predicted_angles.items():
        ref   = reference_angles.get(joint, 0.0)
        error = ref - predicted                        # deviation from normal gait
        kp    = KP.get(joint, 0.5)
        raw   = kp * error
        torques[joint] = float(np.clip(raw, -TORQUE_LIMIT, TORQUE_LIMIT))
        errors[joint]  = float(error)
    return torques, errors

def run(train_pkl: Path, test_pkl: Path, max_chunks: int,
        activity_filter: str, log_csv: bool, make_plots: bool):


    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_regressor.py first.")
    model        = joblib.load(MODEL_PATH)
    target_names = json.loads(TARGET_NAMES_PATH.read_text())
    print(f"Model loaded | {model.n_features_in_} features, {len(target_names)} targets\n")

  
    reference_trajectory = build_reference_trajectory(
        train_pkl, target_names, activity_filter
    )


    if not test_pkl.exists():
        raise FileNotFoundError(f"Test data not found: {test_pkl}")
    raw_df = pd.read_pickle(test_pkl)

    if activity_filter:
        raw_df = raw_df[raw_df["Activity"] == activity_filter].copy()
        print(f"Test data filtered to '{activity_filter}': {len(raw_df):,} rows")

    ta_cols  = [c for c in raw_df.columns if "EMG_TA" in c]
    gm_cols  = [c for c in raw_df.columns if "EMG_GM" in c]
    imu_cols = [c for c in raw_df.columns if any(s in c for s in ["-f", "-c", "-q", "-h"])]

    sensor_data = raw_df[ta_cols + gm_cols + imu_cols].to_numpy(dtype=np.float32)

    ta_idx  = list(range(len(ta_cols)))
    gm_idx  = list(range(len(ta_cols), len(ta_cols) + len(gm_cols)))
    imu_idx = list(range(len(ta_cols) + len(gm_cols),
                         len(ta_cols) + len(gm_cols) + len(imu_cols)))

    gait_col = "Gait" if "Gait" in raw_df.columns else "Gait_Pct"
    gait_vals = raw_df[gait_col].values if gait_col in raw_df.columns else None

    n_chunks = min(len(sensor_data) // CHUNK_SAMPLES, max_chunks)
    print(f"\nRunning {n_chunks} chunks ({CHUNK_SIZE_MS} ms / {CHUNK_SAMPLES} samples each)\n")

    print(f"{'Chunk':>6} | {'Gait%':>6} | {'Total(µs)':>10} | "
          f"{'DA_X pred':>10} | {'DA_X ref':>9} | {'DA_X err':>9} | {'DA_X torq':>10}")
    print("-" * 80)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_rows = []
    timing   = {"feat": [], "infer": [], "total": []}

    for i in range(n_chunks):
        chunk_start = i * CHUNK_SAMPLES
        chunk       = sensor_data[chunk_start : chunk_start + CHUNK_SAMPLES]


        if gait_vals is not None:
            gait_pct = int(np.clip(round(float(gait_vals[chunk_start + CHUNK_SAMPLES - 1])), 0, 100))
        else:
            gait_pct = 0

  
        ref_angles = {tgt: float(reference_trajectory[tgt][gait_pct])
                      for tgt in target_names}


        t0    = time.perf_counter()
        feats = extract_features(chunk, ta_idx, gm_idx, imu_idx)
        t1    = time.perf_counter()
        pred  = model.predict(feats.reshape(1, -1))[0]
        t2    = time.perf_counter()

        feat_us  = (t1 - t0) * 1e6
        infer_us = (t2 - t1) * 1e6
        total_us = (t2 - t0) * 1e6
        timing["feat"].append(feat_us)
        timing["infer"].append(infer_us)
        timing["total"].append(total_us)

        predicted_angles = {tgt: float(pred[j]) for j, tgt in enumerate(target_names)}
        torques, errors  = compute_torques(predicted_angles, ref_angles)

     
        da = "G_DA_X_Target"
        print(f"{i+1:>6} | {gait_pct:>5}% | {total_us:>10.1f} | "
              f"{predicted_angles[da]:>+10.2f}° | {ref_angles[da]:>+8.2f}° | "
              f"{errors[da]:>+8.2f}° | {torques[da]:>+9.2f} Nm")

        row = {"chunk": i+1, "gait_pct": gait_pct,
               "feat_us": round(feat_us, 2), "infer_us": round(infer_us, 2),
               "total_us": round(total_us, 2)}
        for tgt in target_names:
            row[f"pred_{tgt}"]   = round(predicted_angles[tgt], 4)
            row[f"ref_{tgt}"]    = round(ref_angles[tgt], 4)
            row[f"error_{tgt}"]  = round(errors[tgt], 4)
            row[f"torque_{tgt}"] = round(torques[tgt], 4)
        log_rows.append(row)


    print("\n" + "=" * 80)
    print(f"  Timing summary ({n_chunks} chunks)")
    print(f"  {'Stage':<22} {'Avg':>10} {'Min':>10} {'Max':>10}  (µs)")
    print(f"  {'-'*54}")
    for stage, label in [("feat", "Feature extraction"),
                          ("infer", "RF inference"),
                          ("total", "Total")]:
        arr = timing[stage]
        print(f"  {label:<22} {np.mean(arr):>10.1f} {np.min(arr):>10.1f} {np.max(arr):>10.1f}")
    budget_us = CHUNK_SIZE_MS * 1000
    avg_total = np.mean(timing["total"])
    print(f"\n  Chunk budget: {budget_us:,} µs  |  Avg used: {avg_total:.1f} µs  "
          f"({avg_total/budget_us*100:.2f}% of budget)")
    print("=" * 80)

   
    if log_csv and log_rows:
        csv_path = LOG_DIR / "control_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nLog saved: {csv_path}")


    if make_plots and log_rows:
        _make_plots(log_rows, target_names, n_chunks)


def _make_plots(log_rows, target_names, n_chunks):
    chunks = [r["chunk"] for r in log_rows]

 
    n_joints = len(target_names)
    ncols = 3
    nrows = (n_joints + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows), sharex=True)
    axes = axes.flatten()

    for j, tgt in enumerate(target_names):
        ax = axes[j]
        pred_vals  = [r[f"pred_{tgt}"]  for r in log_rows]
        ref_vals   = [r[f"ref_{tgt}"]   for r in log_rows]
        error_vals = [r[f"error_{tgt}"] for r in log_rows]
        torq_vals  = [r[f"torque_{tgt}"] for r in log_rows]

        ax.plot(chunks, ref_vals,   color="green",    lw=1.5, label="Reference (normal gait)")
        ax.plot(chunks, pred_vals,  color="#1f77b4",  lw=1.5, label="Predicted (actual)")
        ax.fill_between(chunks, pred_vals, ref_vals,
                        alpha=0.15, color="red", label="Error zone")

        ax2 = ax.twinx()
        ax2.plot(chunks, torq_vals, color="#ff7f0e", lw=1.2, ls="--", label="Torque (Nm)")
        ax2.set_ylabel("Torque (Nm)", color="#ff7f0e", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        label = tgt.replace("_Target", "").replace("G_", "").replace("_", " ")
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Angle (°)")
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    for j in range(n_joints, len(axes)):
        axes[j].set_visible(False)
    for ax in axes[n_joints - ncols: n_joints]:
        ax.set_xlabel("Chunk")

    fig.suptitle("Reference vs Predicted Joint Angles + Assistive Torque",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(LOG_DIR / "control_reference_vs_predicted.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: control_reference_vs_predicted.png")


    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = axes.flatten()
    gait_pcts = [r["gait_pct"] for r in log_rows]

    for j, tgt in enumerate(target_names):
        ax = axes[j]
        pred_vals = [r[f"pred_{tgt}"] for r in log_rows]
        ref_vals  = [r[f"ref_{tgt}"]  for r in log_rows]
        ax.scatter(gait_pcts, pred_vals, s=10, alpha=0.4, color="#1f77b4", label="Predicted")
        ax.scatter(gait_pcts, ref_vals,  s=10, alpha=0.4, color="green",   label="Reference")
        label = tgt.replace("_Target", "").replace("G_", "").replace("_", " ")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Gait %"); ax.set_ylabel("Angle (°)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for j in range(n_joints, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Predicted vs Reference — plotted over Gait Cycle",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(LOG_DIR / "control_gait_cycle.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: control_gait_cycle.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(chunks, [r["total_us"]  for r in log_rows], label="Total",      lw=1.5)
    ax.plot(chunks, [r["infer_us"]  for r in log_rows], label="Inference",  lw=1.2)
    ax.plot(chunks, [r["feat_us"]   for r in log_rows], label="Features",   lw=1.2)
    ax.axhline(CHUNK_SIZE_MS * 1000, color="red", ls="--", lw=1,
               label=f"Budget ({CHUNK_SIZE_MS} ms)")
    ax.set_xlabel("Chunk"); ax.set_ylabel("Time (µs)")
    ax.set_title("Control Loop Timing per Chunk")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(LOG_DIR / "control_timing.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: control_timing.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pkl",  type=Path, default=DATA_DIR / "training_dataset.pkl")
    parser.add_argument("--test-pkl",   type=Path, default=DATA_DIR / "testing_dataset.pkl")
    parser.add_argument("--max-chunks", type=int,  default=500,
                        help="Max chunks to process (0 = all)")
    parser.add_argument("--activity",   type=str,  default=None,
                        help="Filter to one activity e.g. 'walk'")
    parser.add_argument("--no-csv",     action="store_true")
    parser.add_argument("--no-plots",   action="store_true")
    args = parser.parse_args()

    run(
        train_pkl       = args.train_pkl,
        test_pkl        = args.test_pkl,
        max_chunks      = args.max_chunks if args.max_chunks > 0 else 10**9,
        activity_filter = args.activity,
        log_csv         = not args.no_csv,
        make_plots      = not args.no_plots,
    )


if __name__ == "__main__":
    main()