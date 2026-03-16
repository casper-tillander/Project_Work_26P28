# Project_Work_26P28

## Abstract

The project centers on implementing and optimizing machine learning models for deployment
on NRF5340-DK microcontrollers, aiming to enhance real-time human movement predictions
necessary for exoskeleton applications. Project will explore linear and non-linear models for
human activity classification and ankle kinematics regression using biomechanical signals. The
project offers insights into model performance under strict timing constraints, focusing on
prediction accuracy, inference latency, and memory usage. Students will gain specialist skills in
optimizing AI models for resource-constrained platforms and perform comparative analyses
using a custom dataset to evaluate the feasibility and efficiency of embedded AI solutions in
wearable robotic technology.

## Dataset Description
### Recorded Activities

The dataset captures multiple repetitions of common daily movements:

- **Locomotion**: Walking, Stepping up, and Stepping down.
    
- **Transitions**: Standing up, Sitting down, and Squatting.
    
- **Labeling**: Each activity is labeled with an "Action percentage" (0–100%) to represent the progress of the movement or gait cycle.

### Sensor Configuration & Data Channels

Sampling rate: 2048Hz
The system records a total of **158 signal channels**:

- **EMG (Muscle Activity)**: 2×64 channels (128 total) focusing on the **Tibialis Anterior (TA)** and **Gastrocnemius Medialis (GM)** muscles.

- **IMUs (Motion)**: 4 sensors (24 channels total) placed on the **foot, calf, quad, and hip**. Each IMU provides 3-axis acceleration and 3-axis gyroscope data.
	
- **Goniometers (Joint Angles)**: 3 stations (3×2 channels) measuring the **Ankle (Dominant/Non-Dominant)** and **Knee (Dominant)**.
