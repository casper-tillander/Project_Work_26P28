# Joint angle prediction pipeline for lower-limb assistance

This project was developed as part of the course ELEC-E8004 - Project Work at Aalto University, under the instruction of Ivan Vujaklija.

The system implements a real-time hardware-in-the-loop (HIL) pipeline for predicting ankle joint angles using machine learning. It processes high-density electromyography (EMG) and inertial measurement unit (IMU) data on an nRF5340 microcontroller to perform low-latency inference.

## System overview

<img width="752" height="922" alt="methods" src="https://github.com/user-attachments/assets/05f772d3-7bac-4cf2-bc26-50f9edd70a5b" />

The system utilizes a modular pipeline to deploy random forest regression for real-time joint angle estimation on embedded hardware.

* **Model development**: Random forest models are trained in Python and automatically exported as C-compatible data structures.
* **Embedded execution**: The nRF5340 microcontroller performs real-time feature extraction and inference within a Zephyr RTOS environment.
* **Performance validation**: A hardware-in-the-loop interface enables low-latency data streaming and real-time visualization of prediction accuracy.

## Data specifications

The system utilizes a high-density sensor suite sampled at **2048 Hz**:

*   **EMG (Muscle activity)**: 128 channels from the Tibialis Anterior (TA) and Gastrocnemius Medialis (GM).
*   **IMUs (Motion)**: 24 channels (4 sensors x 6-axis) placed on the foot, calf, quad, and hip.
*   **Goniometers (Ground truth)**: 6 channels measuring ankle and knee angles.

## Project structure

*   `control_system_UART/`: Implementation using standard UART for HIL communication.
*   `control_system_USB/`: Implementation using USB CDC ACM for high-speed HIL communication.
*   `embedded/`: Zephyr RTOS source code, including the regressor logic and feature extraction.
*   `python/`: Data processing, model training, C-code generation, and real-time visualization scripts.

## Building and flashing

The embedded firmware must be built and flashed to the nRF5340-DK before running the Python streaming scripts.

1.  Open the `embedded` folder of your chosen implementation (UART or USB) in VS Code with the nRF Connect Extension.
2.  Create a build configuration for the `nrf5340dk_nrf5340_cpuapp`.
3.  Build the project to generate the binary.
4.  Connect the nRF5340-DK and flash the firmware using the "Flash" button in the extension or via `west flash` in the terminal.

## Pipeline workflow

To deploy and test the system, follow this sequence, or run the provided setup-scripts.

1.  **`split_database.py`**: Partitions raw data into training and testing sets.
2.  **`train_regressor.py`**: Trains the random forest model.
3.  **`export_regressor.py`**: Generates the C files for the MCU. **Note**: If you change the model, you must re-flash the hardware with the new generated files.
4.  **`stream_data.py`**: Once the hardware is flashed and connected, run this script to stream test vectors and visualize the prediction.

## Results

Below are time-series plots comparing the true ankle angles against the predicted values for different movement patterns.

<img width="1108" height="1280" alt="results" src="https://github.com/user-attachments/assets/8af14add-9d08-4a8a-a7e7-b51dc201e99e" />

## Requirements

*   **Hardware**: nRF5340-DK.
*   **Software**:
    *   Python 3.x (pandas, scikit-learn, joblib, pyserial).
    *   nRF Connect SDK (Zephyr RTOS) for embedded development.
