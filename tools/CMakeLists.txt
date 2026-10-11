# 注：img_tools.cpp 用 OpenCV 符号，但 tools 本体不链 OpenCV——
# 由最终可执行文件链 ${OpenCV_LIBS} 时解析（工程内所有消费方都已链接）
add_library(tools STATIC
    logger.cpp
    exiter.cpp
    math_tools.cpp
    crc.cpp
    img_tools.cpp
)

target_link_libraries(tools PUBLIC spdlog::spdlog fmt::fmt yaml-cpp)
