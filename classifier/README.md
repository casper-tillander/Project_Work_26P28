# Movement Classification Pipeline

This folder contains the implementation for classifying physical activities using EMG and IMU data.

## Overview
The classification pipeline is designed to identify the type of movement being performed (e.g., walking, running) before joint angle estimation. This allows for activity-specific regression models.

## Directory Structure
- `data/`: Contains the training and testing datasets.
- `python/`: Scripts for data splitting, feature extraction, and model training.
- `outputs_classifier/`: Results from model experiments, including trained models and confusion matrices.

## Workflow
1. **Data Preparation**: Ensure the raw filtered `.pkl` files are in the `data/` directory.
2. **Data Splitting**: Use `python/split_database_flexible.py` to prepare the datasets.
3. **Model Training**: Run `python/classifier.py` to compare models and export the best one.
4. **Evaluation**: Check the `outputs_classifier/` directory for the model performance metrics and confusion matrices.
