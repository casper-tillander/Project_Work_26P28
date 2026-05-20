import serial
import struct
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from pathlib import Path

COM_PORT = "COM16"
BAUD_RATE = 1000000

# How many predictions to show on the screen at once (Sliding Window)
MAX_PLOT_POINTS = 150

# Thread-safe memory to share data between the Serial stream and the Plot
plot_steps = deque(maxlen=MAX_PLOT_POINTS)
plot_true = deque(maxlen=MAX_PLOT_POINTS)
plot_pred = deque(maxlen=MAX_PLOT_POINTS)
plot_lat = deque(maxlen=MAX_PLOT_POINTS)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

print("Loading test dataset...")
df = pd.read_pickle(DATA_DIR / "testing_dataset.pkl")

ta_cols = [c for c in df.columns if "EMG_TA" in c]
gm_cols = [c for c in df.columns if "EMG_GM" in c]
imu_cols = [c for c in df.columns if any(s in c for s in ["-f", "-c", "-q", "-h"])]
sensor_cols = ta_cols + gm_cols + imu_cols

# Dynamically find the True Angle column (starts with 'G_')
true_angle_col = [c for c in df.columns if c.startswith("G_")][0]

df_sensors = df[sensor_cols].values

# Set academic formatting standards for the live plot
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 100, 
})

# Setup the Plot Figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1.5]})
fig.suptitle(f"Real-Time Hardware-in-the-Loop Inference ({true_angle_col})", weight='bold', fontsize=14)

line_true, = ax1.plot([], [], label='True Angle', color='#2C3E50', linewidth=2)
line_pred, = ax1.plot([], [], label='Predicted Angle', color='#C0392B', linestyle='--', linewidth=2)
ax1.set_ylabel("Angle (deg)")
ax1.legend(loc="upper right", frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

line_lat, = ax2.plot([], [], label='Latency (ms)', color='#7F8C8D', linewidth=1.5, alpha=0.8)
ax2.set_ylabel("Latency (ms)")
ax2.set_xlabel("Prediction Step")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.fill_between([], [], color='#7F8C8D', alpha=0.1) 

def serial_worker():
    print(f"Connecting to {COM_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"Failed to connect to Serial: {e}")
        return

    current_row = 0
    step_counter = 0
    last_send_time = 0

    print("Streaming binary data to microcontroller...")

    while current_row < len(df_sensors):
        raw_line = ser.readline()
        if not raw_line:
            continue
            
        try:
            line = raw_line.decode('utf-8', errors='ignore').strip()
        except:
            continue
        
        if line.startswith("READY_FOR_ROW"):
            if current_row >= len(df_sensors):
                break
                
            row = df_sensors[current_row]
            binary_data = struct.pack(f'<{len(row)}f', *row)
            
            # Start the stopwatch right before sending the bytes
            last_send_time = time.perf_counter()
            ser.write(binary_data)
            current_row += 1
                
        elif line.startswith("PREDICTION"):
            # Stop the stopwatch the exact millisecond the prediction arrives
            arrival_time = time.perf_counter()
            latency_ms = (arrival_time - last_send_time) * 1000.0
            
            try:
                angle = float(line.split(":")[1])
                true_angle = df[true_angle_col].iloc[current_row - 1]
                
                plot_steps.append(step_counter)
                plot_true.append(true_angle)
                plot_pred.append(angle)
                plot_lat.append(latency_ms)
                
                step_counter += 1
            except:
                pass

    ser.close()
    print("Dataset streaming finished.")

def update_plot(frame):
    if len(plot_steps) < 2:
        return line_true, line_pred, line_lat

    # Convert deques to lists for matplotlib
    x_data = list(plot_steps)
    y_true = list(plot_true)
    y_pred = list(plot_pred)
    y_lat = list(plot_lat)

    # Update Top Graph (Angles)
    line_true.set_data(x_data, y_true)
    line_pred.set_data(x_data, y_pred)
    
    # Dynamically scale the axes to create a scrolling effect
    ax1.set_xlim(x_data[0], x_data[-1])
    min_y = min(min(y_true), min(y_pred)) - 5
    max_y = max(max(y_true), max(y_pred)) + 5
    ax1.set_ylim(min_y, max_y)

    # Update Bottom Graph (Latency)
    line_lat.set_data(x_data, y_lat)
    ax2.set_xlim(x_data[0], x_data[-1])
    ax2.set_ylim(0, max(y_lat) + 10) 

    return line_true, line_pred, line_lat

# Start the Serial background thread (daemon=True means it closes when the plot window closes)
thread = threading.Thread(target=serial_worker, daemon=True)
thread.start()

# Start the Live Animation loop (Updates every 30ms)
ani = FuncAnimation(fig, update_plot, interval=30, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()