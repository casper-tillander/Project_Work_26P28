"""
train_regressor.py — updated
Changes vs original:
  1. window_size / step_size aligned to MCU chunk (307 samples = 150 ms @ 2048 Hz)
  2. Per-movement evaluation on test set (separate R² / RMSE per Activity)
  3. Scatter plots + summary bar charts saved to outputs_regressor/plots/

Everything else (split logic, paths, model, save format) is identical to original.
"""

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── Paths (unchanged) ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "outputs_regressor"

META_COLS = ["Activity", "Gait_Pct"]

# ── Window constants — must match main.c ─────────────────────────────────────
# MCU: SAMPLE_RATE_HZ=2048, CHUNK_SIZE_MS=150 → CHUNK_SAMPLES = 307
# Training uses the same size and NO overlap so feature distributions match.
SAMPLE_RATE_HZ = 2048
CHUNK_SIZE_MS  = 150
WINDOW_SIZE    = (SAMPLE_RATE_HZ * CHUNK_SIZE_MS) // 1000   # 307
STEP_SIZE      = WINDOW_SIZE                                 # 0% overlap


def extract_features(raw_df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Converts raw continuous data into sliding window features.
    Window = 307 samples (150 ms) — aligned with MCU chunk size.
    Step   = 307 samples          — no overlap, matches MCU non-overlapping stream.
    """
    print(f"Extracting features | window={window_size} samples "
          f"({window_size/SAMPLE_RATE_HZ*1000:.0f} ms), "
          f"step={step_size} samples "
          f"(overlap={(1-step_size/window_size)*100:.0f}%)")

    ta_cols    = [c for c in raw_df.columns if "EMG_TA" in c]
    gm_cols    = [c for c in raw_df.columns if "EMG_GM" in c]
    imu_cols   = [c for c in raw_df.columns if any(s in c for s in ["-f", "-c", "-q", "-h"])]
    gonio_cols = [c for c in raw_df.columns if c.startswith("G_")]

    rows = []
    for start in range(0, len(raw_df) - window_size + 1, step_size):
        end    = start + window_size
        window = raw_df.iloc[start:end]

        # Keep windows inside a single repetition and activity (unchanged)
        if window['Activity'].nunique() > 1 or window['Reps'].nunique() > 1:
            continue

        last_row = window.iloc[-1]
        feat = {
            "Activity": last_row["Activity"],
            "Gait_Pct": last_row["Gait"]
        }

        feat["EMG_TA_MAV"] = np.mean(np.abs(window[ta_cols].to_numpy()))
        feat["EMG_GM_MAV"] = np.mean(np.abs(window[gm_cols].to_numpy()))
        for col in imu_cols:
            feat[f"IMU_{col}_Mean"] = window[col].mean()
        for col in gonio_cols:
            feat[f"{col}_Target"] = last_row[col]

        rows.append(feat)

    df_out = pd.DataFrame(rows)
    print(f"  → {len(df_out):,} windows extracted")
    return df_out


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
            "features":    X[i].astype(float).tolist(),
            "true_values": np.atleast_1d(y_true[i]).astype(float).tolist(),
            "pred_values": np.atleast_1d(y_pred[i]).astype(float).tolist(),
        })
    return vectors


# ── NEW: per-movement evaluation + plots ─────────────────────────────────────

def evaluate_per_movement(test_df, X_test, y_test, y_pred, targets, plot_dir):
    """
    Computes R² and RMSE separately for each Activity in the test set.
    Saves one scatter plot per activity + two summary bar charts.
    Returns { activity: { target: {r2, rmse, n} } }
    """
    activities   = sorted(test_df["Activity"].unique())
    short_labels = [t.replace("_Target", "").replace("G_", "").replace("_", " ")
                    for t in targets]
    colors       = plt.cm.tab10(np.linspace(0, 0.9, len(targets)))
    results      = {}

    for act in activities:
        mask   = test_df["Activity"].values == act
        yt_act = y_test[mask]
        yp_act = y_pred[mask]
        n      = mask.sum()

        act_res = {}
        for j, tgt in enumerate(targets):
            r2   = r2_score(yt_act[:, j], yp_act[:, j]) if n > 1 else float("nan")
            rmse = np.sqrt(mean_squared_error(yt_act[:, j], yp_act[:, j]))
            act_res[tgt] = {"r2": round(float(r2), 4), "rmse": round(float(rmse), 4), "n": int(n)}
        results[act] = act_res

        # Scatter: true vs predicted for all targets
        ncols = min(3, len(targets))
        nrows = (len(targets) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
        fig.suptitle(f"Activity: {act}  (n={n} windows)", fontsize=13, fontweight="bold")

        for j, (tgt, label) in enumerate(zip(targets, short_labels)):
            ax = axes[j // ncols][j % ncols]
            ax.scatter(yt_act[:, j], yp_act[:, j], alpha=0.4, s=16, color="#1f77b4")
            lims = [min(yt_act[:, j].min(), yp_act[:, j].min()) - 1,
                    max(yt_act[:, j].max(), yp_act[:, j].max()) + 1]
            ax.plot(lims, lims, "r--", lw=1.2, label="Ideal")
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel("True (°)"); ax.set_ylabel("Predicted (°)")
            ax.set_title(f"{label}\nR²={act_res[tgt]['r2']:.3f}  RMSE={act_res[tgt]['rmse']:.2f}°")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        for j in range(len(targets), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        plt.tight_layout()
        fig.savefig(plot_dir / f"scatter_{act.replace(' ', '_')}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Summary bar charts
    x     = np.arange(len(activities))
    width = 0.8 / len(targets)

    for metric, ylabel, fname in [("r2",   "R²",       "r2_per_movement_summary.png"),
                                   ("rmse", "RMSE (°)", "rmse_per_movement_summary.png")]:
        fig, ax = plt.subplots(figsize=(max(10, len(activities) * 1.5), 5))
        for j, (tgt, label) in enumerate(zip(targets, short_labels)):
            vals = [results[act][tgt][metric] for act in activities]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width,
                   label=label, color=colors[j], alpha=0.85)
        if metric == "r2":
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylim(-1, 1)
        ax.set_xlabel("Activity"); ax.set_ylabel(ylabel)
        ax.set_title(f"Per-Movement {ylabel} — Test Set")
        ax.set_xticks(x); ax.set_xticklabels(activities, rotation=20, ha="right")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(plot_dir / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Print table to terminal
    print(f"\n{'Activity':<20}", end="")
    for lbl in short_labels:
        print(f"  {lbl[:12]:<12}", end="")
    print()
    print("-" * (20 + 14 * len(targets)))
    for act in activities:
        print(f"{act:<20}", end="")
        for tgt in targets:
            r = results[act][tgt]
            print(f"  R²={r['r2']:+.2f} {r['rmse']:.1f}°", end="")
        print()

    return results


# ── Main (structure unchanged, additions only) ────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",           nargs="*", default=None)
    parser.add_argument("--n-estimators",     type=int,  default=10)
    parser.add_argument("--max-depth",        type=int,  default=5)
    parser.add_argument("--min-samples-leaf", type=int,  default=10)
    parser.add_argument("--random-state",     type=int,  default=42)
    args = parser.parse_args()

    train_path = DATA_DIR / "training_dataset.pkl"
    test_path  = DATA_DIR / "testing_dataset.pkl"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find training or testing datasets in {DATA_DIR}. "
            "Please run split_database.py first."
        )

    print(f"Loading raw train data: {train_path}")
    raw_train_df = pd.read_pickle(train_path)

    print(f"Loading raw test data:  {test_path}")
    raw_test_df = pd.read_pickle(test_path)

    # Extract features
    train_df = extract_features(raw_train_df)
    test_df  = extract_features(raw_test_df)

    # Organize columns
    targets  = target_columns(train_df, args.target)
    features = feature_columns(train_df, targets)

    # Verify column match between train and test
    missing_features = [c for c in features if c not in test_df.columns]
    missing_targets  = [c for c in targets  if c not in test_df.columns]
    if missing_features or missing_targets:
        raise ValueError(
            f"Train/test columns do not match. "
            f"Missing features: {missing_features}, missing targets: {missing_targets}"
        )

    # Convert to NumPy arrays
    X_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df[targets].to_numpy(dtype=np.float32)
    X_test  = test_df[features].to_numpy(dtype=np.float32)
    y_test  = test_df[targets].to_numpy(dtype=np.float32)

    # Train model (unchanged)
    model = RandomForestRegressor(
        n_estimators    =args.n_estimators,
        max_depth       =args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state    =args.random_state,
        n_jobs=-1,
    )

    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    # Overall evaluation (unchanged output format)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    print(f"Regressor Train R2: {r2_score(y_train, pred_train):.4f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_train, pred_train)):.4f}")
    print(f"Regressor Test R2:  {r2_score(y_test,  pred_test ):.4f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test,  pred_test )):.4f}")
    print(f"Features used: {len(features)}")
    print(f"Targets predicted: {len(targets)}")

    # Per-movement evaluation + plots (NEW)
    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPer-movement evaluation → {plot_dir}")
    per_movement = evaluate_per_movement(
        test_df, X_test, y_test, pred_test, targets, plot_dir
    )

    # Save outputs (unchanged + one new file)
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "regressor_model.pkl")
    (OUTPUT_DIR / "regressor_feature_names.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "regressor_target_names.json").write_text(
        json.dumps(targets, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "regressor_test_vectors.json").write_text(
        json.dumps(make_test_vectors(X_test, y_test, pred_test), indent=2), encoding="utf-8")
    (OUTPUT_DIR / "per_movement_results.json").write_text(
        json.dumps(per_movement, indent=2), encoding="utf-8")

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
