/*
Implementation for the exported Random Forest regressor
- Calls rf_regressor_predict()
- Used in desktop C tests and microcontroller
*/
#include "regressor.h"

void predict_joint_outputs(
    const float features[RF_REG_FEATURE_COUNT],
    float outputs[RF_REG_OUTPUT_COUNT]
)
{
    rf_regressor_predict(features, outputs);
}
