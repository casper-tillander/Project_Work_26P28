@echo off

echo 1. Splitting database...
python split_database.py

echo 2. Training regressor...
python train_regressor.py

echo 3. Exporting regressor...
python export_regressor.py

echo 4. Exporting test data...
python export_test_data.py

echo Done.
pause