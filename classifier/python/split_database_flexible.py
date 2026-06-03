import pandas as pd
from pathlib import Path
import argparse

def split_dataset(subject_id="09", train_ratio=0.8, include_weighted=False):
    # Set up paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent / "data"
    FILE_PATH = DATA_DIR / f"{subject_id}_filtered_all_data.pkl"
    
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Could not find data at {FILE_PATH}")
        
    print(f"Loading data from: {FILE_PATH}")
    df = pd.read_pickle(FILE_PATH)
    
    if not include_weighted:
        print("Filtering out weighted movements...")
        df = df[~df['Activity'].str.endswith('_')].copy()
    else:
        print("Including all movements (including weighted)...")
    
    train_list = []
    test_list = []
    
    # Stratified split by repetitions
    for activity in df['Activity'].unique():
        act_df = df[df['Activity'] == activity]
        reps = sorted(act_df['Reps'].unique()) 
        split_point = int(len(reps) * train_ratio)
        
        train_reps = reps[:split_point]
        test_reps = reps[split_point:]
        
        train_list.append(act_df[act_df['Reps'].isin(train_reps)])
        test_list.append(act_df[act_df['Reps'].isin(test_reps)])
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    suffix = "_all" if include_weighted else ""
    TRAIN_SAVE_PATH = DATA_DIR / f"training_dataset{suffix}.pkl"
    TEST_SAVE_PATH = DATA_DIR / f"testing_dataset{suffix}.pkl"
    
    train_df.to_pickle(TRAIN_SAVE_PATH)
    test_df.to_pickle(TEST_SAVE_PATH)
    
    print(f"Datasets saved to: {TRAIN_SAVE_PATH} and {TEST_SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-weighted", action="store_true")
    args = parser.parse_args()
    
    # Always split without weighted first to ensure consistency with original split
    split_dataset(include_weighted=False)
    # Then split with weighted if requested
    if args.include_weighted:
        split_dataset(include_weighted=True)
