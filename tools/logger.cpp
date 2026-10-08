#include "logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>

namespace tools
{
std::shared_ptr<spdlog::logger> logger()
{
  static std::shared_ptr<spdlog::logger> logger_ = [] {
    auto l = spdlog::stdout_color_mt("rm_vision");
    l->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    l->set_level(spdlog::level::info);  // 调试：改成 spdlog::level::debug
    l->flush_on(spdlog::level::warn);
    return l;
  }();
  return logger_;
}
}  // namespace tools
