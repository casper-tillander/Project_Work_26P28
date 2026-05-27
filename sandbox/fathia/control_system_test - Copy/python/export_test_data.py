import pandas as pd
from pathlib import Path

# Adjust paths if needed
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
df = pd.read_pickle(DATA_DIR / "testing_dataset.pkl")

# Grab only the sensor columns (no metadata or targets)
ta_cols = [c for c in df.columns if "EMG_TA" in c]
gm_cols = [c for c in df.columns if "EMG_GM" in c]
imu_cols = [c for c in df.columns if any(s in c for s in ["-f", "-c", "-q", "-h"])]

sensor_cols = ta_cols + gm_cols + imu_cols
MAX_SAMPLES = 350 
df_sensors = df[sensor_cols].values[:MAX_SAMPLES]

# Write out to a C header
header_path =  BASE_DIR / "embedded" / "src" / "test_data.h"
with open(header_path, "w") as f:
    f.write("#ifndef TEST_DATA_H\n#define TEST_DATA_H\n\n")
    f.write(f"#define NUM_SAMPLES {len(df_sensors)}\n")
    f.write(f"#define NUM_TA_COLS {len(ta_cols)}\n")
    f.write(f"#define NUM_GM_COLS {len(gm_cols)}\n")
    f.write(f"#define NUM_IMU_COLS {len(imu_cols)}\n")
    f.write(f"#define TOTAL_SENSOR_COLS {len(sensor_cols)}\n\n")
    
    f.write("static const float raw_sensor_data[NUM_SAMPLES][TOTAL_SENSOR_COLS] = {\n")
    for row in df_sensors:
        row_str = ", ".join(f"{val:.5f}f" for val in row)
        f.write(f"    {{{row_str}}},\n")
    f.write("};\n\n#endif // TEST_DATA_H\n")

print(f"Generated test_data.h with {len(df_sensors)} samples.")