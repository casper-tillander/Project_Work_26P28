$TRAIN_DATA_PATH = "C:\Users\ansku\Desktop\School\PROJECT\testdata\09_train_features.pkl"
$TEST_DATA_PATH  = "C:\Users\ansku\Desktop\School\PROJECT\testdata\09_test_features.pkl" # add your own data path
$GCC = "C:\msys64\ucrt64\bin\gcc.exe"

python python/train_regressor.py --train-data $TRAIN_DATA_PATH --test-data $TEST_DATA_PATH
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/export_regressor.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/export_regressor_test_vectors.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location c

& $GCC -Wall -Wextra main_regressor_test.c regressor.c rf_regressor_model.c -o reg_test.exe
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    exit $LASTEXITCODE
}

.\reg_test.exe
$TEST_EXIT_CODE = $LASTEXITCODE

Pop-Location
exit $TEST_EXIT_CODE