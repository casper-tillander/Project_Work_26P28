/*
- Compares C predictions against Python predictions.
- only testing, not for micro controlleer
*/

#include <stdio.h>
#include <math.h>
#include "regressor.h"
#include "regressor_test_vectors.h"

#define ERROR_TOLERANCE 0.001f

int main(void)
{
    if (REG_TEST_FEATURE_COUNT != RF_REG_FEATURE_COUNT) {
        printf("ERROR: feature count mismatch: test=%d, model=%d\n",
               REG_TEST_FEATURE_COUNT, RF_REG_FEATURE_COUNT);
        return 1;
    }

    if (REG_TEST_OUTPUT_COUNT != RF_REG_OUTPUT_COUNT) {
        printf("ERROR: output count mismatch: test=%d, model=%d\n",
               REG_TEST_OUTPUT_COUNT, RF_REG_OUTPUT_COUNT);
        return 1;
    }

    int pass_count = 0;

    for (int i = 0; i < REG_TEST_VECTOR_COUNT; ++i) {
        float outputs[RF_REG_OUTPUT_COUNT];
        predict_joint_outputs(reg_test_features[i], outputs);

        int passed = 1;

        printf("Test %d:\n", i);

        for (int o = 0; o < RF_REG_OUTPUT_COUNT; ++o) {
            float expected = reg_expected_outputs[i][o];
            float error = fabsf(outputs[o] - expected);

            if (error > ERROR_TOLERANCE) {
                passed = 0;
            }

            printf("  output[%d]: predicted=%f, expected=%f, error=%f\n",
                   o, outputs[o], expected, error);
        }

        printf("  -> %s\n", passed ? "PASS" : "FAIL");

        if (passed) {
            pass_count++;
        }
    }

    printf("\nSummary: %d/%d tests passed\n", pass_count, REG_TEST_VECTOR_COUNT);

    return (pass_count == REG_TEST_VECTOR_COUNT) ? 0 : 1;
}
