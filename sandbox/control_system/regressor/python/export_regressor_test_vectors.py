"""
- creates C test vectors for regressor.
- Reads outputs_regressor/regressor_test_vectors.json
- converts Python test vectors to C arrays.
- Writes c/regressor_test_vectors.h.
- for testing, not micro controller
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_VECTOR_PATH = PROJECT_ROOT / "outputs_regressor" / "regressor_test_vectors.json"
HEADER_PATH = PROJECT_ROOT / "c" / "regressor_test_vectors.h"


def c_float(value) -> str:
    text = format(float(value), ".9g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def main() -> None:
    if not TEST_VECTOR_PATH.exists():
        raise FileNotFoundError(f"Missing test vector file: {TEST_VECTOR_PATH}")

    vectors = json.loads(TEST_VECTOR_PATH.read_text(encoding="utf-8"))

    if not vectors:
        raise ValueError("regressor_test_vectors.json is empty")

    vector_count = len(vectors)
    feature_count = len(vectors[0]["features"])
    output_count = len(vectors[0]["pred_values"])

    lines = []
    lines.append("#ifndef REGRESSOR_TEST_VECTORS_H")
    lines.append("#define REGRESSOR_TEST_VECTORS_H")
    lines.append("")
    lines.append(f"#define REG_TEST_VECTOR_COUNT {vector_count}")
    lines.append(f"#define REG_TEST_FEATURE_COUNT {feature_count}")
    lines.append(f"#define REG_TEST_OUTPUT_COUNT {output_count}")
    lines.append("")

    lines.append(
        "static const float reg_test_features"
        "[REG_TEST_VECTOR_COUNT][REG_TEST_FEATURE_COUNT] = {"
    )
    for item in vectors:
        values = ", ".join(c_float(v) for v in item["features"])
        lines.append(f"    {{{values}}},")
    lines.append("};")
    lines.append("")

    lines.append(
        "static const float reg_expected_outputs"
        "[REG_TEST_VECTOR_COUNT][REG_TEST_OUTPUT_COUNT] = {"
    )
    for item in vectors:
        values = ", ".join(c_float(v) for v in item["pred_values"])
        lines.append(f"    {{{values}}},")
    lines.append("};")
    lines.append("")

    lines.append(
        "static const float reg_true_outputs"
        "[REG_TEST_VECTOR_COUNT][REG_TEST_OUTPUT_COUNT] = {"
    )
    for item in vectors:
        values = ", ".join(c_float(v) for v in item["true_values"])
        lines.append(f"    {{{values}}},")
    lines.append("};")
    lines.append("")

    lines.append("#endif")
    lines.append("")

    HEADER_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {HEADER_PATH}")
    print(f"Vectors: {vector_count}")
    print(f"Features per vector: {feature_count}")
    print(f"Outputs per vector: {output_count}")


if __name__ == "__main__":
    main()
