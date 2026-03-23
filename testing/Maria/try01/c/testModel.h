#ifndef TESTMODEL_H
#define TESTMODEL_H

#define TREE0_NODE_COUNT 63
#define RF_FEATURE_COUNT 78
#define RF_CLASS_COUNT 5

extern const int tree0_features[TREE0_NODE_COUNT];
extern const float tree0_thresholds[TREE0_NODE_COUNT];
extern const int tree0_children_left[TREE0_NODE_COUNT];
extern const int tree0_children_right[TREE0_NODE_COUNT];
extern const int tree0_leaf_class[TREE0_NODE_COUNT];
int estimate(const float features[RF_FEATURE_COUNT]);

#endif