import pandas as pd
from pathlib import Path

def split_dataset(subject_id="09", train_ratio=0.8):
    # Set up paths (matches your original directory structure)
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent / "data"
    FILE_PATH = DATA_DIR / f"{subject_id}_filtered_all_data.pkl"
    
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Could not find data at {FILE_PATH}")
        
    print(f"Loading data from: {FILE_PATH}")
    df = pd.read_pickle(FILE_PATH)
    
    # Filter out activities ending with '_' (weighted variants)
    df = df[~df['Activity'].str.endswith('_')].copy()
    
    train_list = []
    test_list = []
    
    # Stratified split by repetitions
    for activity in df['Activity'].unique():
        act_df = df[df['Activity'] == activity]
        
        # Get unique reps and sort them to ensure consistent chronological splitting
        reps = sorted(act_df['Reps'].unique()) 
        
        # Calculate where to split (80% mark)
        split_point = int(len(reps) * train_ratio)
        
        train_reps = reps[:split_point]
        test_reps = reps[split_point:]
        
        train_list.append(act_df[act_df['Reps'].isin(train_reps)])
        test_list.append(act_df[act_df['Reps'].isin(test_reps)])
    
    # Combine lists back into solid DataFrames
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    # Check the distribution of activities in each set
    print("\nDistribution in train and test set:")
    print(f"{'Activity':<20} | {'Train reps':<25} | {'Test reps':<20}")
    print("-" * 70)
    
    for act in sorted(df['Activity'].unique()):
        # Count the unique reps
        train_count = train_df[train_df['Activity'] == act]['Reps'].nunique()
        test_count = test_df[test_df['Activity'] == act]['Reps'].nunique()
        
        # Print the counts
        print(f"{str(act):<15} | {train_count:<20} | {test_count}")
    
    # Save the files
    # Ensure the output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    TRAIN_SAVE_PATH = DATA_DIR / "training_dataset.pkl"
    TEST_SAVE_PATH = DATA_DIR / "testing_dataset.pkl"
    
    train_df.to_pickle(TRAIN_SAVE_PATH)
    test_df.to_pickle(TEST_SAVE_PATH)
    
    print(f"\nDatasets saved to:")
    print(f"- {TRAIN_SAVE_PATH}")
    print(f"- {TEST_SAVE_PATH}")

if __name__ == "__main__":
    split_dataset()