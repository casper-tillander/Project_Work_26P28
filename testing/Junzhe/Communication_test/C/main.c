#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <stdio.h>
#include <string.h> // For memmove()
#include <math.h>   // For sqrt()

#include "model_regressor.h" 

// --- CONFIGURATION ---
#define WINDOW_SIZE 200
#define STEP_SIZE 100
#define OVERLAP (WINDOW_SIZE - STEP_SIZE) 

#define NUM_CHANNELS 24   // 24 IMU channels
#define NUM_FEATURES 48   // 24 Means + 24 RMS values
#define PACKET_SIZE (NUM_CHANNELS * sizeof(float)) // 96 bytes

// --- GLOBAL BUFFERS ---
// 1. Sliding Window Buffer
float sensor_buffer[WINDOW_SIZE][NUM_CHANNELS];
int current_row = 0; 

// 2. Feature & ANN Buffers
float extracted_features[NUM_FEATURES]; 
float scaled_features[NUM_FEATURES];
float hidden_layer[NUM_HIDDEN];
float predicted_angle[NUM_TARGETS]; 

// 3. UART Interrupt Buffers (Aligned for floats)
float rx_buf_float[NUM_CHANNELS]; 
uint8_t *rx_buf = (uint8_t *)rx_buf_float; 
int rx_ptr = 0;
bool packet_ready = false;

const struct device *uart_dev = DEVICE_DT_GET(DT_NODELABEL(uart0));

// ---------------------------------------------------------
// 1. FEATURE EXTRACTION FUNCTION
// ---------------------------------------------------------
void extract_features() {
    for (int c = 0; c < NUM_CHANNELS; c++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int r = 0; r < WINDOW_SIZE; r++) {
            float val = sensor_buffer[r][c];
            sum += val;
            sum_sq += (val * val); 
        }
        
        extracted_features[c] = sum / WINDOW_SIZE;               // Mean
        extracted_features[c + NUM_CHANNELS] = sqrt(sum_sq / WINDOW_SIZE); // RMS
    }
}

// ---------------------------------------------------------
// 2. THE NEURAL NETWORK INFERENCE FUNCTION
// ---------------------------------------------------------
void predict_joint_angle(const float* raw_features) {
    // Step A: Standard Scaler
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (raw_features[i] - scaler_mean[i]) / scaler_scale[i];
    }

    // Step B: Hidden Layer Math & ReLU Activation
    for (int i = 0; i < NUM_HIDDEN; i++) {
        hidden_layer[i] = b1[i]; 
        for (int j = 0; j < NUM_FEATURES; j++) {
            hidden_layer[i] += scaled_features[j] * W1[i][j];
        }
        if (hidden_layer[i] < 0.0f) {
            hidden_layer[i] = 0.0f;
        }
    }

    // Step C: Output Layer Math 
    for (int i = 0; i < NUM_TARGETS; i++) {
        predicted_angle[i] = b2[i]; 
        for (int j = 0; j < NUM_HIDDEN; j++) {
            predicted_angle[i] += hidden_layer[j] * W2[i][j];
        }
    }
}

// ---------------------------------------------------------
// 3. UART INTERRUPT CALLBACK
// ---------------------------------------------------------
void uart_cb(const struct device *dev, void *user_data) {
    if (!uart_irq_update(dev)) return;

    if (uart_irq_rx_ready(dev)) {
        uint8_t c;
        while (uart_fifo_read(dev, &c, 1) == 1) {
            rx_buf[rx_ptr++] = c;
            if (rx_ptr == PACKET_SIZE) {
                packet_ready = true;
                rx_ptr = 0; 
            }
        }
    }
}

// ---------------------------------------------------------
// 4. MAIN EXECUTION LOOP
// ---------------------------------------------------------
int main(void) {
    if (!device_is_ready(uart_dev)) {
        printf("UART device not found!\n");
        return 0;
    }

    uart_irq_callback_set(uart_dev, uart_cb);
    uart_irq_rx_enable(uart_dev);

    printf("nRF5340 HIL Continuous Stream Initialized.\n");
    printf("Waiting for Python to stream raw IMU data...\n");

    while (1) {
        if (packet_ready) {
            
            // 1. Cast raw bytes back to 24 floats
            float *incoming_pc_row = (float *)rx_buf;
            
            // 2. Add to Circular Buffer
            for(int c = 0; c < NUM_CHANNELS; c++) {
                sensor_buffer[current_row][c] = incoming_pc_row[c];
            }
            current_row++;

            // 3. Check if Window is Full
            if(current_row == WINDOW_SIZE) {
                
                // --- THE MACHINE LEARNING PIPELINE ---
                extract_features();
                predict_joint_angle(extracted_features);
                
                // Print the predictions back to the PC! 
                // Assuming Index 0 is the Dominant Ankle (G_DA_X)
                // Multiply by 1000 (e.g., 14.523 becomes 14523)
                int angle_scaled = (int)(predicted_angle[0] * 1000.0f);
                
                // (Axis 0)
                printf("Pred Angle: %d\n", angle_scaled);
                
                // 4. Slide the window 100 steps
                memmove(sensor_buffer[0], 
                        sensor_buffer[STEP_SIZE], 
                        OVERLAP * NUM_CHANNELS * sizeof(float));
                
                current_row = OVERLAP; 
            }

            packet_ready = false; 
        }
        
        k_msleep(1); 
    }
    return 0;
}