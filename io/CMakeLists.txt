find_library(GXIAPI_LIB NAMES gxiapi REQUIRED)

add_library(io STATIC
    camera.cpp
    galaxy/galaxy.cpp
)

# io/ 本身作为包含目录：其它模块可写 #include "camera.hpp" / "galaxy/galaxy.hpp"
target_include_directories(io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(io PUBLIC tools ${OpenCV_LIBS} ${GXIAPI_LIB} pthread)
