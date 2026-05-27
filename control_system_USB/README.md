# USB-based hardware-in-the-loop pipeline

This directory contains the implementation of the prediction pipeline using the native USB peripheral of the nRF5340 for high-speed data streaming.

## Setup and requirements
- **Hardware switch**: Set the nRF5340-DK power switch to **USB**.
- **Connection**: Requires **two micro USB cables**:
  - One connected to the **J-Link port** (for flashing and debugging).
  - One connected to the **nRF USB port** (for data streaming via CDC ACM).

## Execution sequence
1. **Python pipeline**: Run `python/run_python_pipeline.bat` (or the `.sh` equivalent) to process data, train the model, and export the C source files.
2. **Flash hardware**: Build and flash the `embedded` project to the nRF5340-DK.
3. **Data streaming**: Run `python/stream_data.py` to begin the real-time HIL simulation.

## Difference from UART pipeline
This version utilizes the nRF5340 native USB controller as a CDC ACM device, providing higher data throughput for the HIL simulation. This makes the demo run faster, but the prediction latency is slower.
