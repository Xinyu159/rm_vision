# 任务层（检测/解算/跟踪/决策）。当前只有 auto_aim/ 的检测部分（armor + yolo）
find_library(NVINFER_LIB NAMES nvinfer REQUIRED)
find_library(CUDART_LIB NAMES cudart PATHS /usr/local/cuda/lib64 REQUIRED)

add_library(auto_aim STATIC
    auto_aim/armor.cpp
    auto_aim/yolo.cpp
)

# tasks/ 本身作为包含目录：模块内可写 #include "auto_aim/yolo.hpp"
target_include_directories(auto_aim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# TensorRT 头文件在系统路径（/usr/include/aarch64-linux-gnu/NvInfer.h），
# CUDA 头文件（cuda_runtime.h）需显式指定
if(EXISTS /usr/local/cuda/include/cuda_runtime.h)
  target_include_directories(auto_aim PUBLIC /usr/local/cuda/include)
endif()

target_link_libraries(auto_aim PUBLIC tools ${OpenCV_LIBS} ${NVINFER_LIB} ${CUDART_LIB})

# TRT 8.5 起经典绑定 API（getBindingDimensions/enqueueV2 等）标记 deprecated，
# 本实现有意沿用（两个静态绑定，够用且简单）；要迁移到 Tensor V3 API 时去掉这行
target_compile_options(auto_aim PRIVATE -Wno-deprecated-declarations)
