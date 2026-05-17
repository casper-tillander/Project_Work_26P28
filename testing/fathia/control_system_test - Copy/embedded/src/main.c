#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <math.h>
#include "regressor.h"
#include "test_data.h"

/*
 * Window / Chunk alignment
 * ─────────────────────────────────────────────────────────────────────────
 * SAMPLE_RATE_HZ : 2048 Hz
 * CHUNK_SIZE_MS  : 150 ms  →  CHUNK_SAMPLES = 2048 × 0.150 = 307 samples
 *
 * MUST match train_regressor.py:
 *   WINDOW_SIZE = (2048 * 150) // 1000 = 307  ✓
 *   STEP_SIZE   = 307  (no overlap)            ✓
 *
 * The MCU processes data in non-overlapping 150 ms chunks, consuming each
 * sample exactly once. The RF was trained on windows of the same size and
 * stride, so the feature distribution at inference matches training.
 * ─────────────────────────────────────────────────────────────────────────
 */
#define SAMPLE_RATE_HZ  2048
#define CHUNK_SIZE_MS   150
#define CHUNK_SAMPLES   ((SAMPLE_RATE_HZ * CHUNK_SIZE_MS) / 1000)   /* = 307 */

#if CHUNK_SAMPLES != 307
#warning "CHUNK_SAMPLES changed — update WINDOW_SIZE in train_regressor.py to match!"
#endif

// Prints a float without printf (sign + whole.frac)
static void print_float_3(const char *name, float value)
{
    float pos_val = (value < 0.0f) ? -value : value;
    int whole = (int)pos_val;
    int frac  = (int)((pos_val - whole) * 1000.0f);
    printk("%s%s%d.%03d", name, (value < 0.0f ? "-" : ""), whole, frac);
}

// Extract features from one non-overlapping 150 ms chunk.
// Feature order must match training exactly:
//   [0]   EMG_TA_MAV
//   [1]   EMG_GM_MAV
//   [2…]  IMU mean per channel (24 channels: f/c/q/h × accel/gyro × 3 axes)
void extract_features(const float chunk[CHUNK_SAMPLES][TOTAL_SENSOR_COLS],
                      float *features_out)
{
    int f_idx = 0;

    // EMG TA MAV
    float ta_sum = 0.0f;
    for (int i = 0; i < CHUNK_SAMPLES; i++)
        for (int c = 0; c < NUM_TA_COLS; c++)
            ta_sum += fabsf(chunk[i][c]);
    features_out[f_idx++] = ta_sum / (CHUNK_SAMPLES * NUM_TA_COLS);

    // EMG GM MAV
    float gm_sum = 0.0f;
    for (int i = 0; i < CHUNK_SAMPLES; i++)
        for (int c = 0; c < NUM_GM_COLS; c++)
            gm_sum += fabsf(chunk[i][NUM_TA_COLS + c]);
    features_out[f_idx++] = gm_sum / (CHUNK_SAMPLES * NUM_GM_COLS);

    // IMU means
    for (int c = 0; c < NUM_IMU_COLS; c++) {
        float sum = 0.0f;
        for (int i = 0; i < CHUNK_SAMPLES; i++)
            sum += chunk[i][NUM_TA_COLS + NUM_GM_COLS + c];
        features_out[f_idx++] = sum / CHUNK_SAMPLES;
    }
}

// ── Simple proportional torque controller ────────────────────────────────────
// Mirrors control_system.py compute_torques() on the PC side.
// Kp values (Nm/deg) and neutral angles — tune for your actuator.
static const float KP[RF_REG_OUTPUT_COUNT]      = {0.5f, 0.2f, 0.8f, 0.2f, 0.5f, 0.2f};
static const float NEUTRAL[RF_REG_OUTPUT_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#define TORQUE_LIMIT 20.0f

static float clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static void compute_torques(const float angles[RF_REG_OUTPUT_COUNT],
                             float torques[RF_REG_OUTPUT_COUNT])
{
    for (int j = 0; j < RF_REG_OUTPUT_COUNT; j++) {
        float raw = KP[j] * (angles[j] - NEUTRAL[j]);
        torques[j] = clampf(raw, -TORQUE_LIMIT, TORQUE_LIMIT);
    }
}

int main(void)
{
    printk("nRF5340 Exoskeleton Control Demo\n");
    printk("Rate: %d Hz | Chunk: %d ms (%d samples)\n\n",
           SAMPLE_RATE_HZ, CHUNK_SIZE_MS, CHUNK_SAMPLES);

    int num_chunks = NUM_SAMPLES / CHUNK_SAMPLES;
    printk("Loaded %d samples → %d chunks\n\n", NUM_SAMPLES, num_chunks);

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int start_row = chunk_idx * CHUNK_SAMPLES;

        uint32_t t0 = k_cycle_get_32();

        // 1. Feature extraction
        float features[RF_REG_FEATURE_COUNT];
        extract_features(&raw_sensor_data[start_row], features);

        // 2. Inference
        float angles[RF_REG_OUTPUT_COUNT];
        predict_joint_outputs(features, angles);

        // 3. Control
        float torques[RF_REG_OUTPUT_COUNT];
        compute_torques(angles, torques);

        uint32_t time_us = k_cyc_to_us_floor32(k_cycle_get_32() - t0);

        // 4. Output
        printk("Chunk %3d | ", chunk_idx + 1);
        print_float_3("DA_X=", angles[0]);
        printk("deg ");
        print_float_3("torque=", torques[0]);
        printk("Nm | %u us\n", time_us);

        k_sleep(K_MSEC(CHUNK_SIZE_MS));
    }

    printk("\nDemo complete.\n");
    while (1) { k_sleep(K_SECONDS(1)); }
    return 0;
}
