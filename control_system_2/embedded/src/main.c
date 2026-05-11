#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <math.h>
#include "regressor.h"
#include "test_data.h"

// Configuration
#define SAMPLE_RATE_HZ 2048
#define CHUNK_SIZE_MS 150
#define CHUNK_SAMPLES ((SAMPLE_RATE_HZ * CHUNK_SIZE_MS) / 1000)

// Prints floats
static void print_float_3(const char *name, float value)
{
    float pos_val = (value < 0.0f) ? -value : value;
    int whole = (int)pos_val;
    int frac = (int)((pos_val - whole) * 1000.0f);
    printk("%s%s%d.%03d", name, (value < 0.0f ? "-" : ""), whole, frac);
}

// Extract features from the data chunk
void extract_features(const float chunk[CHUNK_SAMPLES][TOTAL_SENSOR_COLS], float *features_out) {
    int f_idx = 0;

    // Calculate EMG TA MAV
    float ta_sum = 0.0f;
    for(int i = 0; i < CHUNK_SAMPLES; i++) {
        for(int c = 0; c < NUM_TA_COLS; c++) {
            ta_sum += fabsf(chunk[i][c]);
        }
    }
    features_out[f_idx++] = ta_sum / (CHUNK_SAMPLES * NUM_TA_COLS);

    // Calculate EMG GM MAV
    float gm_sum = 0.0f;
    for(int i = 0; i < CHUNK_SAMPLES; i++) {
        for(int c = 0; c < NUM_GM_COLS; c++) {
            gm_sum += fabsf(chunk[i][NUM_TA_COLS + c]);
        }
    }
    features_out[f_idx++] = gm_sum / (CHUNK_SAMPLES * NUM_GM_COLS);

    // Calculate IMU Means
    for(int c = 0; c < NUM_IMU_COLS; c++) {
        float sum = 0.0f;
        for(int i = 0; i < CHUNK_SAMPLES; i++) {
            sum += chunk[i][NUM_TA_COLS + NUM_GM_COLS + c];
        }
        features_out[f_idx++] = sum / CHUNK_SAMPLES;
    }
}

int main(void)
{
    printk("nRF5340 Regressor Embedded Demo\n");
    printk("Loaded %d raw samples. Window size: %d ms.\n\n", NUM_SAMPLES, CHUNK_SIZE_MS);

    int num_chunks = NUM_SAMPLES / CHUNK_SAMPLES;

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        // Calculate where the chunk starts
        int start_row = chunk_idx * CHUNK_SAMPLES;

        // Start timer
        uint32_t start_cycles = k_cycle_get_32();

        // Extract Features from the chunk
        float current_features[RF_REG_FEATURE_COUNT];
        extract_features(&raw_sensor_data[start_row], current_features);

        // Run the model
        float outputs[RF_REG_OUTPUT_COUNT];
        predict_joint_outputs(current_features, outputs);

        // Stop timer and calculate processing time
        uint32_t end_cycles = k_cycle_get_32();
        uint32_t diff_cycles = end_cycles - start_cycles;
        uint32_t time_us = k_cyc_to_us_floor32(diff_cycles); // Convert to microseconds!

        // Output the results
        printk("Chunk %3d | ", chunk_idx + 1);
        
        print_float_3("Predicted Ankle Angle: ", outputs[0]); 
        printk(" deg | Processing Time: %u us\n", time_us);

        // Sleep for 150ms
        k_sleep(K_MSEC(CHUNK_SIZE_MS)); 
    }

    printk("\nDemo Sequence Complete.\n");

    while (1) {
        k_sleep(K_SECONDS(1));
    }

    return 0;
}