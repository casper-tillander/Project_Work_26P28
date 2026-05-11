# Project_Work_26P28

Project directory for a machine learning influenced control system to be run on a NRF5340-DK microcontroller.

## Dataset Description

### Sensor Configuration & Data Channels

Sampling rate: **2048Hz***
The system records a total of **158 signal channels**:

- **EMG (Muscle Activity)**: 2×64 channels (128 total) from the **Tibialis Anterior (TA)** and **Gastrocnemius Medialis (GM)** muscles.

- **IMUs (Motion)**: 4 sensors (24 channels total) placed on the **foot, calf, quad, and hip**. Each IMU provides 3-axis acceleration and 3-axis gyroscope data.
	
- **Goniometers (Joint Angles)**: 3 stations (3×2 channels) measuring the **Ankle (Dominant/Non-Dominant)** and **Knee (Dominant)**.
