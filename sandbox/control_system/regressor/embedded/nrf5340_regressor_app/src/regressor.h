/*
interface for the exported Random Forest regressor
- creates predict_joint_outputs(), a function for using the regresso.
- Used in desktop C tests and on the microcontroller
*/

#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "rf_regressor_model.h"

void predict_joint_outputs(
    const float features[RF_REG_FEATURE_COUNT],
    float outputs[RF_REG_OUTPUT_COUNT]
);

#endif
