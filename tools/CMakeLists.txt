add_library(tools STATIC
    logger.cpp
    exiter.cpp
    math_tools.cpp
)

target_link_libraries(tools PUBLIC spdlog::spdlog fmt::fmt yaml-cpp)
