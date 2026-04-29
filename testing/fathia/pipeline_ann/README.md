The pipeline supports two models:
- ANN (Feedforward Neural Network)
- Random Forest Regressor

In this implementation, the pre-trained ANN weights from `model_regressor.h` are used.

---

## Project Structure

```
project/
├── CMakeLists.txt
├── prj.conf
├── export_test_data.py
└── src/
    ├── main.c
    ├── pipeline_config.h
    ├── test_data.h
    └── model_regressor.h
```

---

## Step 1: Generate Test Data

Before building, run the Python script to export IMU windows from the dataset into a C header file:

```bash
python export_test_data.py
```

This reads `09_filtered_all_data.pkl`, takes sliding windows from the test split (last 20% of data), and exports them as C float arrays into `test_data.h`. Copy the generated file into the `src/` folder.

This only needs to be run once. It replaces the role of physical sensors during benchmarking.

---

## Step 2: Select Model

Open `src/pipeline_config.h` and uncomment the model you want to use:

```c
// #define USE_MODEL_RF
#define USE_MODEL_ANN
```

For ANN, only `model_regressor.h` is needed. For RF, additional source files are required and the corresponding lines in `CMakeLists.txt` must be uncommented.

---

## Step 3: Build and Flash

1. Open the project folder in VS Code with the nRF Connect extension
2. Click "Add build configuration"
3. Select board: `nrf5340dk/nrf5340/cpuapp`
4. Click "Generate and Build"
5. Once built, click "Flash"

---

## Step 4: View Output

1. Connect the board via USB
2. In the nRF Connect panel, expand "Connected Devices"
3. Click on the device and select "Application core" to open the RTT terminal
4. Press the RESET button on the board to restart the program

---

## Expected Output

```
Project 26P28 - Exoskeleton ML Pipeline
Model   : ANN (Feedforward)
Features: 48  Outputs: 6
Windows : 10  Size: 410  Step: 102

Win0 1523us
G_DA_X:p=-11.409 t=-11.181 e=0.228
G_DA_Y:p=-4.715  t=-3.984  e=0.731
[CTRL] Ankle:MAINTAIN

BENCHMARK SUMMARY
Avg latency : 1523 us
G_DA_X  MAE=0.228 deg
G_DA_Y  MAE=0.731 deg
```

---

## Notes

- `test_data.h` contains real IMU data exported from the dataset
- `model_regressor.h` contains the pre-trained ANN weights including scaler parameters
- The pipeline is general and can be switched to RF by changing `pipeline_config.h`
- Flash overflow will occur if the RF model is too large. Retrain with fewer trees and smaller depth before exporting
