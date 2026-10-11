find_library(GXIAPI_LIB NAMES gxiapi REQUIRED)

# vendored 串口库（wjwwood serial，见 io/serial/ 与 README）——须在 add_library(io) 前
add_subdirectory(serial)

add_library(io STATIC
    camera.cpp
    galaxy/galaxy.cpp
    gimbal/gimbal.cpp
)

# io/ 本身作为包含目录：其它模块可写 #include "camera.hpp" / "galaxy/galaxy.hpp"
target_include_directories(io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(io PUBLIC tools ${OpenCV_LIBS} ${GXIAPI_LIB} pthread serial)
