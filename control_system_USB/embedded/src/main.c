#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/usb/usb_device.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "regressor.h"

// Timing and buffer configuration
#define SAMPLE_RATE_HZ 2048
#define CHUNK_SIZE_MS 150
#define CHUNK_SAMPLES ((SAMPLE_RATE_HZ * CHUNK_SIZE_MS) / 1000)
#define STEP_SAMPLES (CHUNK_SAMPLES / 2)
#define KEEP_SAMPLES (CHUNK_SAMPLES - STEP_SAMPLES)

// Sensor and model configuration
#define NUM_TA_COLS 64
#define NUM_GM_COLS 64
#define NUM_IMU_COLS 24
#define TOTAL_SENSOR_COLS 152

#define RF_REG_FEATURE_COUNT 26
#define RF_REG_OUTPUT_COUNT 6 

// Global memory
float window[CHUNK_SAMPLES][TOTAL_SENSOR_COLS];

// Binary UART reader
void read_exact_bytes(const struct device *uart, uint8_t *data, size_t len) {
    size_t rx_bytes = 0;
    while (rx_bytes < len) {
        unsigned char c;
        if (uart_poll_in(uart, &c) == 0) {
            data[rx_bytes++] = c;
        } else {
            k_yield();
        }
    }
}

// Helper function to print floats
static void print_float_3(const char *name, float value)
{
    float pos_val = (value < 0.0f) ? -value : value;
    int whole = (int)pos_val;
    int frac = (int)((pos_val - whole) * 1000.0f);
    printk("%s%s%d.%03d", name, (value < 0.0f ? "-" : ""), whole, frac);
}

// Feature extraction
void extract_features(const float chunk[CHUNK_SAMPLES][TOTAL_SENSOR_COLS], float *features_out) {
    int f_idx = 0;

    float ta_sum = 0.0f;
    for(int i = 0; i < CHUNK_SAMPLES; i++) {
        for(int c = 0; c < NUM_TA_COLS; c++) {
            ta_sum += fabsf(chunk[i][c]);
        }
    }
    features_out[f_idx++] = ta_sum / (CHUNK_SAMPLES * NUM_TA_COLS);

    float gm_sum = 0.0f;
    for(int i = 0; i < CHUNK_SAMPLES; i++) {
        for(int c = 0; c < NUM_GM_COLS; c++) {
            gm_sum += fabsf(chunk[i][NUM_TA_COLS + c]);
        }
    }
    features_out[f_idx++] = gm_sum / (CHUNK_SAMPLES * NUM_GM_COLS);

    for(int c = 0; c < NUM_IMU_COLS; c++) {
        float sum = 0.0f;
        for(int i = 0; i < CHUNK_SAMPLES; i++) {
            sum += chunk[i][NUM_TA_COLS + NUM_GM_COLS + c];
        }
        features_out[f_idx++] = sum / CHUNK_SAMPLES;
    }
}

int main(void) {
    const struct device *uart_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
    if (!device_is_ready(uart_dev)) {
        return 0;
    }

    // Enable the native USB peripheral
    if (usb_enable(NULL)) {
        return 0;
    }

    // Wait for the Python script to connect via USB
    uint32_t dtr = 0;
    while (true) {
        uart_line_ctrl_get(uart_dev, UART_LINE_CTRL_DTR, &dtr);
        if (dtr) {
            break;
        }
        k_sleep(K_MSEC(100));
    }

    k_sleep(K_MSEC(100));

    unsigned char dummy;
    while (uart_poll_in(uart_dev, &dummy) == 0) {}

    printk("nRF5340 HIL Binary Inference Started\n");

    int samples_needed = CHUNK_SAMPLES;
    int write_idx = 0;

    while (1) {
        printk("READY_FOR_CHUNK:%d\n", samples_needed);
        
        size_t bytes_to_read = samples_needed * TOTAL_SENSOR_COLS * sizeof(float);
        
        // Read the block
        read_exact_bytes(uart_dev, (uint8_t *)window[write_idx], bytes_to_read);

        k_sleep(K_MSEC(1));

        // Start timer
        uint32_t start_cycles = k_cycle_get_32();

        // Extract and predict
        float current_features[RF_REG_FEATURE_COUNT];
        extract_features(window, current_features);
        
        float outputs[RF_REG_OUTPUT_COUNT];
        predict_joint_outputs(current_features, outputs);

        // Stop timer and calculate latency
        uint32_t end_cycles = k_cycle_get_32();
        uint32_t hw_latency_us = k_cyc_to_us_floor32(end_cycles - start_cycles);

        // Print prediction and latency
        print_float_3("PREDICTION:", outputs[0]);
        printk(",LATENCY_US:%u\n", hw_latency_us);

        // Shift buffer
        memmove(window, window + STEP_SAMPLES, KEEP_SAMPLES * sizeof(window[0]));
        
        write_idx = KEEP_SAMPLES;
        samples_needed = STEP_SAMPLES; 
    }

    return 0;
}