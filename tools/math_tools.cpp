#include "math_tools.hpp"

namespace tools
{
double delta_time(
  std::chrono::steady_clock::time_point b, std::chrono::steady_clock::time_point a)
{
  std::chrono::duration<double> d = b - a;
  return d.count();
}
}  // namespace tools
