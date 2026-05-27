"""
- exports the trained regressor into C code
- loads outputs_regressor/regressor_model.pkl
- extracts all tree structures and leaf output values
- writes c/rf_regressor_model.h, c/rf_regressor_model.c
- generates C files for microcontroller
"""

from pathlib import Path
import json
import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "outputs_regressor" / "regressor_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR.parent / "outputs_regressor" / "regressor_feature_names.json"
TARGET_NAMES_PATH = BASE_DIR.parent / "outputs_regressor" / "regressor_target_names.json"

# Output paths
C_DIR = BASE_DIR.parent / "embedded" / "src"
HEADER_PATH = C_DIR / "rf_regressor_model.h"
SOURCE_PATH = C_DIR / "rf_regressor_model.c"


def format_c_value(value, ctype: str) -> str:
    """Format Python values as valid C literals."""
    if ctype == "float":
        v = float(value)
        text = format(v, ".9g")
        if "e" not in text and "." not in text:
            text += ".0"
        return text + "f"

    return str(int(value))


def c_array(name: str, values: list, ctype: str, per_line: int = 8) -> str:
    lines = [f"const {ctype} {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        formatted = ", ".join(format_c_value(v, ctype) for v in chunk)
        lines.append(f"    {formatted},")
    lines.append("};")
    return "\n".join(lines)


def c_2d_array(name: str, rows: list[list[float]], ctype: str, cols: int) -> str:
    lines = [f"const {ctype} {name}[{len(rows)}][{cols}] = {{"]
    for row in rows:
        formatted = ", ".join(format_c_value(v, ctype) for v in row)
        lines.append(f"    {{{formatted}}},")
    lines.append("};")
    return "\n".join(lines)


def get_leaf_output(tree, node_index: int, output_count: int) -> list[float]:
    """
    For sklearn RandomForestRegressor:
    tree.value[node] shape is usually (n_outputs, 1).
    For a single-output regressor, it is still handled as one output.
    """
    value = np.asarray(tree.value[node_index], dtype=float)

    if value.ndim == 2:
        outputs = value[:, 0]
    elif value.ndim == 1:
        outputs = value
    else:
        outputs = value.reshape(-1)

    outputs = outputs[:output_count]

    if len(outputs) != output_count:
        raise ValueError(
            f"Leaf output count mismatch: expected {output_count}, got {len(outputs)}"
        )

    return [float(v) for v in outputs]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    target_names = json.loads(TARGET_NAMES_PATH.read_text(encoding="utf-8"))

    tree_count = len(model.estimators_)
    feature_count = int(model.n_features_in_)
    output_count = len(target_names)

    tree_offsets = []
    tree_node_counts = []

    all_features = []
    all_thresholds = []
    all_children_left = []
    all_children_right = []
    all_leaf_values = []

    offset = 0

    for estimator in model.estimators_:
        tree = estimator.tree_
        node_count = int(tree.node_count)

        tree_offsets.append(offset)
        tree_node_counts.append(node_count)

        features = tree.feature.tolist()
        thresholds = tree.threshold.tolist()
        children_left = tree.children_left.tolist()
        children_right = tree.children_right.tolist()

        for i in range(node_count):
            if features[i] == -2:
                leaf_values = get_leaf_output(tree, i, output_count)
            else:
                leaf_values = [0.0] * output_count

            all_leaf_values.append(leaf_values)

        # Convert local child indices to global node indices.
        for i in range(node_count):
            if children_left[i] != -1:
                children_left[i] += offset
            if children_right[i] != -1:
                children_right[i] += offset

        all_features.extend(features)
        all_thresholds.extend(thresholds)
        all_children_left.extend(children_left)
        all_children_right.extend(children_right)

        offset += node_count

    total_node_count = offset

    header = f"""#ifndef RF_REGRESSOR_MODEL_H
#define RF_REGRESSOR_MODEL_H

#define RF_REG_TREE_COUNT {tree_count}
#define RF_REG_FEATURE_COUNT {feature_count}
#define RF_REG_OUTPUT_COUNT {output_count}
#define RF_REG_TOTAL_NODE_COUNT {total_node_count}

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
"""

    source = f"""#include "rf_regressor_model.h"

{c_array("rf_reg_tree_offsets", tree_offsets, "int")}
{c_array("rf_reg_tree_node_counts", tree_node_counts, "int")}

{c_array("rf_reg_features", all_features, "int")}
{c_array("rf_reg_thresholds", all_thresholds, "float")}
{c_array("rf_reg_children_left", all_children_left, "int")}
{c_array("rf_reg_children_right", all_children_right, "int")}

{c_2d_array("rf_reg_leaf_values", all_leaf_values, "float", output_count)}

static int predict_tree_leaf(int tree_index, const float features[RF_REG_FEATURE_COUNT])
{{
    int node = rf_reg_tree_offsets[tree_index];

    while (rf_reg_features[node] != -2) {{
        int feat = rf_reg_features[node];
        float threshold = rf_reg_thresholds[node];

        if (features[feat] <= threshold) {{
            node = rf_reg_children_left[node];
        }} else {{
            node = rf_reg_children_right[node];
        }}
    }}

    return node;
}}

void rf_regressor_predict(
    const float features[RF_REG_FEATURE_COUNT],
    float outputs[RF_REG_OUTPUT_COUNT]
)
{{
    for (int o = 0; o < RF_REG_OUTPUT_COUNT; ++o) {{
        outputs[o] = 0.0f;
    }}

    for (int t = 0; t < RF_REG_TREE_COUNT; ++t) {{
        int leaf = predict_tree_leaf(t, features);

        for (int o = 0; o < RF_REG_OUTPUT_COUNT; ++o) {{
            outputs[o] += rf_reg_leaf_values[leaf][o];
        }}
    }}

    for (int o = 0; o < RF_REG_OUTPUT_COUNT; ++o) {{
        outputs[o] /= (float)RF_REG_TREE_COUNT;
    }}
}}
"""

    HEADER_PATH.write_text(header, encoding="utf-8")
    SOURCE_PATH.write_text(source, encoding="utf-8")

    print(f"Wrote: {HEADER_PATH}")
    print(f"Wrote: {SOURCE_PATH}")
    print(f"Trees: {tree_count}")
    print(f"Features: {feature_count}")
    print(f"Outputs: {output_count}")
    print(f"Total nodes: {total_node_count}")


if __name__ == "__main__":
    main()
