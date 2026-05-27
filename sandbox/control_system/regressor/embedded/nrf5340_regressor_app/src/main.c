#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include "regressor.h"

static const float test_features[RF_REG_FEATURE_COUNT] = {
    0.0386584811f, 0.00656342646f, 11.1949482f, -4.15935278f,
    5.4588294f, 0.105655991f, -1.26798165f, -0.990985453f,
    9.52435875f, -1.30470276f, -3.77967286f, 0.958879173f,
    -1.80923891f, 0.263153762f, 10.2901974f, -1.1390233f,
    1.83809769f, 0.857571065f, -0.599149942f, 0.554611385f,
    10.4175186f, -0.664038658f, 0.108892143f, 0.136477426f,
    0.186219394f, -0.133032352f
};

static const float expected_outputs[RF_REG_OUTPUT_COUNT] = {
    -11.4095913f, -4.7155658f, 1.40574084f,
    -7.59042614f, -2.72320478f, 1.14362248f
};

static void print_float_3(const char *name, float value)
{
    int scaled = (int)(value * 1000.0f);
    int whole = scaled / 1000;
    int frac = scaled % 1000;

    if (frac < 0) {
        frac = -frac;
    }

    printk("%s%d.%03d", name, whole, frac);
}

int main(void)
{
    float outputs[RF_REG_OUTPUT_COUNT];

    printk("Regressor embedded test start\n");
    printk("Features: %d, outputs: %d, trees: %d\n",
           RF_REG_FEATURE_COUNT, RF_REG_OUTPUT_COUNT, RF_REG_TREE_COUNT);

    predict_joint_outputs(test_features, outputs);

    for (int i = 0; i < RF_REG_OUTPUT_COUNT; ++i) {
        float error = outputs[i] - expected_outputs[i];
        if (error < 0.0f) {
            error = -error;
        }

        printk("output[%d]: ", i);
        print_float_3("predicted=", outputs[i]);
        printk(", ");
        print_float_3("expected=", expected_outputs[i]);
        printk(", ");
        print_float_3("error=", error);
        printk("\n");
    }

    printk("Regressor embedded test done\n");

    while (1) {
        k_sleep(K_SECONDS(1));
    }

    return 0;
}
