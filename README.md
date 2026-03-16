# Project_Work_26P28

### Recorded Activities

The dataset captures multiple repetitions of common daily movements:

- **Locomotion**: Walking, Stepping up, and Stepping down.
    
- **Transitions**: Standing up, Sitting down, and Squatting.
    
- **Labeling**: Each activity is labeled with an "Action percentage" (0–100%) to represent the progress of the movement or gait cycle.

### 2. Sensor Configuration & Data Channels

Sampling rate: 2048Hz
The system records a total of **158 signal channels**:

- **EMG (Muscle Activity)**: 2×64 channels (128 total) focusing on the **Tibialis Anterior (TA)** and **Gastrocnemius Medialis (GM)** muscles.

[![|120](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Tibialis.png/180px-Tibialis.png)](https://en.wikipedia.org/wiki/File:Tibialis.png)
[![|120](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Sobo_1909_303_-_Gastrocnemius_muscle.png/180px-Sobo_1909_303_-_Gastrocnemius_muscle.png)](https://en.wikipedia.org/wiki/File:Sobo_1909_303_-_Gastrocnemius_muscle.png)

- **IMUs (Motion)**: 4 sensors (24 channels total) placed on the **foot, calf, quad, and hip**. Each IMU provides 3-axis acceleration and 3-axis gyroscope data.
	
- **Goniometers (Joint Angles)**: 3 stations (3×2 channels) measuring the **Ankle (Dominant/Non-Dominant)** and **Knee (Dominant)**.
