cmake_minimum_required(VERSION 3.16)
project(rm_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

# ---------- 大恒 Galaxy SDK ----------
# 安装脚本已把 libgxiapi.so 放进 /usr/lib；头文件在 SDK 解压目录里
set(GALAXY_INC
    "$ENV{HOME}/sdk/galaxy/Galaxy_Linux-arm64_Gige-U3_2.4.2507.8231/Galaxy_Linux-arm64_Gige-U3_2.4.2507.8231/Galaxy_camera/inc"
    CACHE PATH "Galaxy SDK 头文件目录")
if(NOT EXISTS "${GALAXY_INC}/GxIAPI.h")
  message(FATAL_ERROR "找不到 Galaxy SDK 头文件 GxIAPI.h（当前 GALAXY_INC=${GALAXY_INC}）")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${GALAXY_INC})

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks)

#################### src ####################
add_executable(standard src/standard.cpp)
target_link_libraries(standard ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)

#################### tests ####################
add_executable(camera_test tests/camera_test.cpp)
target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)

add_executable(camera_view tests/camera_view.cpp)
target_link_libraries(camera_view ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)

add_executable(detect_view tests/detect_view.cpp)
target_link_libraries(detect_view ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)

add_executable(yolo_test tests/yolo_test.cpp)
target_link_libraries(yolo_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim)

add_executable(solver_test tests/solver_test.cpp)
target_link_libraries(solver_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim)

add_executable(serial_test tests/serial_test.cpp)
target_link_libraries(serial_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
