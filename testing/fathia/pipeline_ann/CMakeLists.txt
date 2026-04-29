cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ml_pipeline_26p28)

# ── Always included ──────────────────────────────────────────────────────────
target_sources(app PRIVATE
    src/main.c
)

# ── RF only: uncomment these when USE_MODEL_RF is set in pipeline_config.h ───
# target_sources(app PRIVATE src/regressor.c)
# target_sources(app PRIVATE src/rf_regressor_model.c)
