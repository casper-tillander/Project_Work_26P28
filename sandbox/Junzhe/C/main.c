#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "model_lda.h"
#include "test_data.h"

int predict_activity(const float *raw_features) {
    float scaled_features[NUM_FEATURES];
    float class_scores[NUM_CLASSES];

    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (raw_features[i] - scaler_mean[i]) / scaler_scale[i];
    }

    for (int c = 0; c < NUM_CLASSES; c++) {
        class_scores[c] = lda_intercept[c];
        for (int f = 0; f < NUM_FEATURES; f++) {
            class_scores[c] += scaled_features[f] * lda_coef[c][f];
        }
    }

    int best_class = 0;
    float max_score = class_scores[0];
    
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (class_scores[c] > max_score) {
            max_score = class_scores[c];
            best_class = c;
        }
    }

    return best_class; 
}

int main(void) {
    k_msleep(2000); // waiting for the serial connection
    
    printk("\n========================================\n");
    printk("Starting NRF5340 LDA Benchmark...\n");
    printk("Model Features: %d | Classes: %d\n", NUM_FEATURES, NUM_CLASSES);
    
    int correct_predictions = 0;
    uint32_t total_time_ns = 0; 

    for(int i = 0; i < NUM_TEST_SAMPLES; i++) {
        
        // 1. start of the recording
        uint32_t start_cycles = k_cycle_get_32();
        
        // 2. execute model inference
        int prediction = predict_activity(test_features[i]);
        
        // 3. record the hardware cycle count at the end
        uint32_t stop_cycles = k_cycle_get_32();

        int actual = test_labels[i];
        if (prediction == actual) {
            correct_predictions++;
        }

        // 4. calculate the number of cycles consumed, and convert to nanoseconds (ns) and microseconds (us)
        uint32_t cycles_spent = stop_cycles - start_cycles;
        uint32_t ns_spent = k_cyc_to_ns_floor32(cycles_spent);
        uint32_t us_spent = ns_spent / 1000;

        total_time_ns += ns_spent;

        // print the time for each inference
        printk("Sample %03d | Pred: %d | Actual: %d | Time: %u us\n", i, prediction, actual, us_spent);
        
        k_msleep(10); // prevent serial port congestion
    }

    // calculate the average inference time
    uint32_t avg_time_ns = total_time_ns / NUM_TEST_SAMPLES;
    
    printk("\n--- Benchmark Complete ---\n");
    printk("Final Accuracy : %d / %d\n", correct_predictions, NUM_TEST_SAMPLES);
	printk("Total Time(100 samples): %u ns (%.3f ms)\n", total_time_ns, (float)total_time_ns / 1000000.0);
    printk("Avg Time/Sample: %u ns (%.3f ms)\n", avg_time_ns, (float)avg_time_ns / 1000000.0);
    printk("========================================\n");
    
    return 0;
}