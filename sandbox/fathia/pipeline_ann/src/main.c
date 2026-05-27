/* Switch models: Change define in pipeline_config.h

1. Load pre-exported test windows from test_data.h
2. Feature extraction: mean + RMS per IMU channel
3. Model inference (RF or ANN)
4. Print: predicted | ground truth | error | latency
5. Final benchmark summary (avg latency, MAE per joint)

 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/timing/timing.h>
#include <math.h>
#include <string.h>

#include "pipeline_config.h"
#include "test_data.h"

#if defined(USE_MODEL_RF)
    #include "regressor.h"
    #define MODEL_NAME       "Random Forest"
    #define MODEL_N_OUTPUTS  RF_REG_OUTPUT_COUNT
    #define MODEL_N_FEATURES RF_REG_FEATURE_COUNT

#elif defined(USE_MODEL_ANN)
    #include "model_regressor.h"
    #define MODEL_NAME       "ANN (Feedforward)"
    #define MODEL_N_OUTPUTS  NUM_TARGETS
    #define MODEL_N_FEATURES NUM_FEATURES

#else
    #error "Set USE_MODEL_RF or USE_MODEL_ANN in pipeline_config.h"
#endif


#if MODEL_N_FEATURES != TEST_N_FEATURES
    #error "Feature count mismatch: model vs test_data.h"
#endif


static const char *joint_names[] = {
    "G_DA_X", "G_DA_Y", "G_DK_X", "G_DK_Y", "G_NDA_X", "G_NDA_Y"
};
#define N_JOINT_NAMES 6


static float features[TEST_N_FEATURES];
static float outputs [MODEL_N_OUTPUTS];

#if defined(USE_MODEL_ANN)
static float scaled_features[NUM_FEATURES];
static float hidden_layer   [NUM_HIDDEN];
#endif


static void extract_features(const float *raw_window)
{
    const int n_ch  = TEST_N_IMU_CH;
    const int n_smp = TEST_WINDOW_SIZE;

    for (int ch = 0; ch < n_ch; ch++) {
        float sum    = 0.0f;
        float sum_sq = 0.0f;
        for (int t = 0; t < n_smp; t++) {
            float v  = raw_window[t * n_ch + ch];
            sum     += v;
            sum_sq  += v * v;
        }
        features[ch]        = sum    / (float)n_smp;
        features[ch + n_ch] = sqrtf(sum_sq / (float)n_smp);
    }
}


static void run_inference(void)
{
#if defined(USE_MODEL_RF)

    predict_joint_outputs(features, outputs);

#elif defined(USE_MODEL_ANN)

  
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
    }

   
    for (int i = 0; i < NUM_HIDDEN; i++) {
        hidden_layer[i] = b1[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            hidden_layer[i] += scaled_features[j] * W1[i][j];
        }
        if (hidden_layer[i] < 0.0f) {
            hidden_layer[i] = 0.0f;
        }
    }


    for (int i = 0; i < NUM_TARGETS; i++) {
        outputs[i] = b2[i];
        for (int j = 0; j < NUM_HIDDEN; j++) {
            outputs[i] += hidden_layer[j] * W2[i][j];
        }
    }

#endif
}

                                                              */
/*  Takes predicted joint angles and decides exoskeleton assistance level.     */
/*  In real deployment this would send commands to motor controllers.          */
/*  Here we simulate the decision and print the control action.                */

static void control_update(const float *angles, int n)
{
    /* Index 0 = G_DA_X (Dominant Ankle Sagittal) */
    /* Index 2 = G_DK_X (Dominant Knee Sagittal)  */
    float ankle = angles[0];
    float knee  = (n > 2) ? angles[2] : 0.0f;

    /* ── Ankle control ── */
    const char *ankle_cmd;
    if (ankle < -15.0f) {
        ankle_cmd = "INCREASE support (dorsiflexion)";
    } else if (ankle > -5.0f) {
        ankle_cmd = "REDUCE support (plantarflexion)";
    } else {
        ankle_cmd = "MAINTAIN (normal range)";
    }

    /* ── Knee control ── */
    const char *knee_cmd;
    if (knee < -10.0f) {
        knee_cmd = "FLEXION assist ON";
    } else if (knee > 5.0f) {
        knee_cmd = "EXTENSION assist ON";
    } else {
        knee_cmd = "NEUTRAL — no assist";
    }

    printk("  [CTRL] Ankle: %s\n", ankle_cmd);
    printk("  [CTRL] Knee : %s\n", knee_cmd);
}


static void pf(const char *label, float v)
{
    int s = (int)(v * 1000.0f);
    int w = s / 1000;
    int f = s % 1000;
    if (f < 0) f = -f;
    printk("%s%d.%03d", label, w, f);
}


{
   
    uint64_t total_latency_us          = 0;
    uint64_t min_latency_us            = UINT64_MAX;
    uint64_t max_latency_us            = 0;
    float    sum_mae[MODEL_N_OUTPUTS];
    float    max_err[MODEL_N_OUTPUTS];

    memset(sum_mae, 0, sizeof(sum_mae));
    memset(max_err, 0, sizeof(max_err));

    printk("\n");
    printk("##############################################\n");
    printk("  Project 26P28 — Exoskeleton ML Pipeline\n");
    printk("  Target  : NRF5340-DK (Cortex-M33 + FPU)\n");
    printk("  Model   : %s\n", MODEL_NAME);
    printk("  Features: %d  Outputs: %d\n",
           MODEL_N_FEATURES, MODEL_N_OUTPUTS);
    printk("  Windows : %d  Size: %d  Step: %d\n",
           TEST_N_WINDOWS, TEST_WINDOW_SIZE, TEST_STEP_SIZE);
    printk("##############################################\n\n");

    timing_init();
    timing_start();

    for (int w = 0; w < TEST_N_WINDOWS; w++) {

        const float *raw = test_imu_windows[w];

        
        timing_t t0 = timing_counter_get();
        extract_features(raw);
        run_inference();
        timing_t t1 = timing_counter_get();

       
        control_update(outputs, MODEL_N_OUTPUTS);

        uint64_t lat = timing_cycles_to_ns(
            timing_cycles_get(&t0, &t1)) / 1000ULL;

        total_latency_us += lat;
        if (lat < min_latency_us) min_latency_us = lat;
        if (lat > max_latency_us) max_latency_us = lat;

        
        printk("Win%d %lluus\n", w, lat);
        k_msleep(30);

        for (int o = 0; o < MODEL_N_OUTPUTS; o++) {
            float pred  = outputs[o];
            float truth = test_true_outputs[w][o];
            float err   = pred - truth;
            if (err < 0.0f) err = -err;

            sum_mae[o] += err;
            if (err > max_err[o]) max_err[o] = err;

            const char *name = (o < N_JOINT_NAMES) ? joint_names[o] : "out";
            printk("%s:", name);
            pf("p=", pred);
            pf(" t=", truth);
            pf(" e=", err);
            printk("\n");
            k_msleep(30);
        }

        float ankle = outputs[0];
        if (ankle < -15.0f)     printk("[CTRL] Ankle:INCREASE\n");
        else if (ankle > -5.0f) printk("[CTRL] Ankle:REDUCE\n");
        else                    printk("[CTRL] Ankle:MAINTAIN\n");
        k_msleep(100);
    }

    timing_stop();

    uint64_t avg_us = total_latency_us / (uint64_t)TEST_N_WINDOWS;

    printk("##############################################\n");
    printk("  BENCHMARK SUMMARY\n");
    printk("  Model  : %s\n", MODEL_NAME);
    printk("##############################################\n");
    printk("  Latency (us):\n");
    printk("    Avg : %llu us  (%llu ms)\n", avg_us, avg_us / 1000ULL);
    printk("    Min : %llu us\n", min_latency_us);
    printk("    Max : %llu us\n", max_latency_us);
    printk("  Accuracy (over %d windows):\n", TEST_N_WINDOWS);
    for (int o = 0; o < MODEL_N_OUTPUTS; o++) {
        float mae = sum_mae[o] / (float)TEST_N_WINDOWS;
        const char *name = (o < N_JOINT_NAMES) ? joint_names[o] : "out";
        printk("    %-8s  MAE=", name);
        pf("", mae);
        printk(" deg  MaxErr=");
        pf("", max_err[o]);
        printk(" deg\n");
    }
    printk("##############################################\n\n");

    while (1) {
        k_sleep(K_SECONDS(1));
    }

    return 0;
}