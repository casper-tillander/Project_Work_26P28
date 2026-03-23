import pickle
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "outputs" / "rf_model.pkl"
EXPORT_PATH = BASE_DIR.parent / "outputs" / "tree0_export.json"

# load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# For simplicity and trial, only exporting the first tree, will change later
tree = model.estimators_[0].tree_

features = tree.feature.tolist()
thresholds = tree.threshold.tolist()
kidsOnLeft = tree.children_left.tolist()
kidsOnRight = tree.children_right.tolist()

leaf_class = []
for i in range(tree.node_count):
    if tree.feature[i] == -2:  # if last node
        cls = int(np.argmax(tree.value[i][0]))
    else:
        cls = -1
    leaf_class.append(cls)

export_data = {
    "node_count": tree.node_count,
    "features": features,
    "thresholds": thresholds,
    "children_left": kidsOnLeft,
    "children_right": kidsOnRight,
    "leaf_class": leaf_class
}

with open(EXPORT_PATH, "w") as f:
    json.dump(export_data, f, indent=2)

print(f"First tree at: {EXPORT_PATH}")