cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf5340_regressor_app)

target_sources(app PRIVATE
    src/main.c
    src/regressor.c
    src/rf_regressor_model.c
)

target_include_directories(app PRIVATE src)
