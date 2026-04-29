# Hardware-In-The-Loop (HIL) Communication Test Pipeline

This repository contains the complete pipeline for training an Artificial Neural Network (ANN) on IMU data, deploying it to an nRF5340 microcontroller, and running a continuous Hardware-In-The-Loop (HIL) serial communication test.

## Project Structure

### 1. `regressor_all_Gonio` (Model Generation)

This script handles the machine learning training phase.

It processes the ground truth goniometer data, trains the ANN Multi-Layer Perceptron (MLP), and extracts the weights and biases.

- **Output:** It automatically exports the trained network into a C-header file (`model_regressor.h`) that is ready to be compiled directly into the microcontroller's flash memory.

### 2. `C code folder` (nRF5340 Firmware)

This directory contains the Zephyr RTOS C-code that runs on the embedded board.

It sets up a UART hardware interrupt, establishes a sliding window (circular buffer) in RAM, calculates features (Mean, RMS), and runs the `model_regressor.h` math.
**(more specific procedure can be found in previous folder)**

- **Setup Procedure:** 
	1. Move the generated `model_regressor.h` file into your `src`  folder.
    
    2. Ensure your `prj.conf` has the necessary UART Interrupt and Floating-Point Unit (FPU) configurations enabled.
    
    3. Perform a Pristine Build and Flash the board.

### 3. `raw_data_exporter` (PC-to-Board Streaming)

This Python script simulates a real-time sensor by streaming raw data to the board and listening for the predictions.

==It loads the `.pkl` dataset, packs the raw IMU floats into tight binary packets (`struct.pack`), and sends them over the USB serial connection. It simultaneously listens for the microcontroller's predicted angle response.==

---

##  CRITICAL WARNING: Serial Port Hijacking 

**DO NOT open the serial port in nRF Connect, VS Code, PuTTY, or any other serial monitor while running the `raw_data_exporter` script!**

A serial port (UART/USB) can only be accessed by **one program at a time**.

If you leave a terminal open in nRF Connect, it will "hijack" the data stream. Your Python script will either crash with an "Access Denied" error, or the board's responses will be eaten by the nRF terminal, causing the Python script to wait endlessly.

**Always close your terminal connections before running the Python streaming script.**

---

## Quick Start Guide

1. **Train & Export:** Run the `regressor_all_Gonio` script to generate your `model_regressor.h` file.
    
2. **Flash the Board:** Place the `.h` file into the C project, build the firmware, and flash it to the nRF5340.
    
3. **Verify Port:** Update the `SERIAL_PORT` variable in the `raw_data_exporter` script to match your specific OS path (e.g., `COM3` for Windows, `/dev/tty.usbmodem...` for Mac/Linux).
    
4. **Close Monitors:** Ensure absolutely no serial monitors are currently connected to the board.
    
5. **Run the Stream:** Execute the `raw_data_exporter` script to begin the real-time HIL simulation. You should see the board replying with its predictions.
