#ifndef TOOLS__EXITER_HPP
#define TOOLS__EXITER_HPP

#include <atomic>

namespace tools
{
/// Ctrl+C / SIGTERM 退出标志（语义对齐 sp_vision_25 的 tools::Exiter）
class Exiter
{
public:
  Exiter();
  bool exit() const;

private:
  static std::atomic<bool> quit_;
  static void handler(int signum);
};
}  // namespace tools

#endif  // TOOLS__EXITER_HPP
