# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0
#
# model_bench — OpenVINO GenAI C++ benchmark suite.
#
# Build:
#   source /opt/intel/openvino_2026/setupvars.sh
#   source /opt/intel/openvino_genai/setupvars.sh
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Optional multimodal image input for the VLM task:
#   cmake -S . -B build -DMB_WITH_OPENCV=ON   # requires OpenCV dev packages
cmake_minimum_required(VERSION 3.16)
project(model_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Resolve OpenVINO GenAI (and OpenVINO via the same cmake dir). setupvars.sh
# exports OpenVINO_DIR / OpenVINOGenAI_DIR; the PATHS fallback mirrors the
# official samples.
find_package(OpenVINOGenAI REQUIRED
    PATHS
        "$ENV{OpenVINOGenAI_DIR}"
        "$ENV{OpenVINO_DIR}"
        "/opt/intel/openvino_genai/runtime/cmake"
    NO_CMAKE_FIND_ROOT_PATH
)

option(MB_WITH_OPENCV "Build VLM image input support via OpenCV" OFF)

find_package(Threads REQUIRED)

file(GLOB COMMON_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")
file(GLOB TASK_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/*.cpp")

add_executable(model_bench
    src/main.cpp
    ${COMMON_SRC}
    ${TASK_SRC}
)

target_include_directories(model_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(model_bench PRIVATE openvino::genai openvino::runtime Threads::Threads)

if(MB_WITH_OPENCV)
    find_package(OpenCV REQUIRED)
    target_compile_definitions(model_bench PRIVATE MB_WITH_OPENCV)
    target_link_libraries(model_bench PRIVATE ${OpenCV_LIBS})
    target_include_directories(model_bench PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

set_target_properties(model_bench PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS model_bench RUNTIME DESTINATION bin)
