# UART-based hardware-in-the-loop pipeline

This directory contains the implementation of the prediction pipeline using standard UART communication for data streaming between the PC and the nRF5340-DK.

## Setup and requirements
- **Hardware switch**: Set the power switch on the nRF5340-DK to **VDD**.
- **Connection**: A single micro USB cable connected to the J-Link port is used for both flashing and UART communication.

## Execution sequence
1. **Python pipeline**: Run `python/run_python_pipeline.bat` (or the `.sh` equivalent) to process data, train the model, and export the C source files.
2. **Flash hardware**: Build and flash the `embedded` project to the nRF5340-DK using the nRF Connect extension.
3. **Data streaming**: Run `python/stream_data.py` to begin the real-time HIL simulation.

## Difference from USB pipeline
This version uses the onboard J-Link debugger's UART bridge. It is simpler to set up but has lower maximum throughput compared to the native USB implementation.
