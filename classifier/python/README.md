# Movement Classification Scripts

This directory contains the Python scripts for training and evaluating movement classification models.

## Scripts

### 1. `split_database_flexible.py`
Partitions the raw filtered data into training and testing datasets. It supports:
- Stratified splitting by repetitions for each activity.
- Optional inclusion of weighted movements (e.g., activities with extra weight).
- Filtering out weighted movements by default to maintain consistency with the regression pipeline.

**Usage:**
```bash
python split_database_flexible.py [--include-weighted]
```

### 2. `classifier.py`
The main training and evaluation script.
- **Feature Extraction**: Converts continuous EMG and IMU data into sliding windows (150ms length, 50% overlap).
  - **EMG Features**: Mean Absolute Value (MAV) for TA and GM channels.
  - **IMU Features**: Mean value for each 6-axis sensor channel.
- **Model Comparison**: Trains and evaluates multiple classifiers:
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
- **Evaluation**: Generates classification reports and confusion matrices.
- **Persistence**: Saves the best performing model, label encoder, and feature names to `outputs_classifier/`.

**Usage:**
```bash
python classifier.py [--suffix _all]
```

## Feature Engineering
The classifier uses a window-based approach to capture temporal dynamics:
- **Window Size**: 150ms (307 samples at 2048 Hz).
- **Step Size**: 75ms (153 samples, 50% overlap).
- **Labels**: Extracted from the `Activity` column of the dataset.
