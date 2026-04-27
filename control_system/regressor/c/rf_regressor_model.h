#ifndef RF_REGRESSOR_MODEL_H
#define RF_REGRESSOR_MODEL_H

#define RF_REG_TREE_COUNT 100
#define RF_REG_FEATURE_COUNT 26
#define RF_REG_OUTPUT_COUNT 6
#define RF_REG_TOTAL_NODE_COUNT 123200

extern const int rf_reg_tree_offsets[RF_REG_TREE_COUNT];
extern const int rf_reg_tree_node_counts[RF_REG_TREE_COUNT];

extern const int rf_reg_features[RF_REG_TOTAL_NODE_COUNT];
extern const float rf_reg_thresholds[RF_REG_TOTAL_NODE_COUNT];
extern const int rf_reg_children_left[RF_REG_TOTAL_NODE_COUNT];
extern const int rf_reg_children_right[RF_REG_TOTAL_NODE_COUNT];
extern const float rf_reg_leaf_values[RF_REG_TOTAL_NODE_COUNT][RF_REG_OUTPUT_COUNT];

void rf_regressor_predict(
    const float features[RF_REG_FEATURE_COUNT],
    float outputs[RF_REG_OUTPUT_COUNT]
);

#endif
