from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = [
    root / "notebooks" / "build_database_reps_separation.ipynb",
    root / "python" / "train_regressor.py",
    root / "python" / "export_regressor.py",
    root / "python" / "export_regressor_test_vectors.py",
    root / "c" / "regressor.h",
    root / "c" / "regressor.c",
    root / "c" / "main_regressor_test.c",
]

for path in files:
    print(("OK  " if path.exists() else "MISS") + str(path))
