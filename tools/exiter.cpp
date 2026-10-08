#include "exiter.hpp"

#include <csignal>

namespace tools
{
std::atomic<bool> Exiter::quit_{false};

Exiter::Exiter()
{
  std::signal(SIGINT, handler);
  std::signal(SIGTERM, handler);
}

bool Exiter::exit() const { return quit_.load(); }

void Exiter::handler(int) { quit_ = true; }
}  // namespace tools
