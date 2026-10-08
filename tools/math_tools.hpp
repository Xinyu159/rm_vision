#ifndef TOOLS__MATH_TOOLS_HPP
#define TOOLS__MATH_TOOLS_HPP

#include <chrono>

namespace tools
{
/// b - a，单位秒（命名对齐 sp_vision_25）
double delta_time(
  std::chrono::steady_clock::time_point b, std::chrono::steady_clock::time_point a);

template <typename T>
T square(T x)
{
  return x * x;
}
}  // namespace tools

#endif  // TOOLS__MATH_TOOLS_HPP
