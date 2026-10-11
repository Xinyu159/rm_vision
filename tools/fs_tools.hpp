#ifndef TOOLS__FS_TOOLS_HPP
#define TOOLS__FS_TOOLS_HPP

// 文件系统小工具（不引 C++17 <filesystem>：GCC 9 下它可能要求额外链库，
// 而本工程 cmake 顶格 3.16 / 工具链固定，手写 POSIX 版最稳）

#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cstddef>
#include <string>

namespace tools
{
/// 递归建目录（等价 mkdir -p）；已存在视为成功
inline bool mkdirs(const std::string & path)
{
  if (path.empty()) return false;

  std::string cur;
  for (std::size_t i = 0; i < path.size(); i++) {
    cur += path[i];
    if (path[i] != '/' && i + 1 != path.size()) continue;

    std::string dir = cur;
    while (!dir.empty() && dir.back() == '/') dir.pop_back();
    if (dir.empty()) continue;  // 根目录 '/'
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) return false;
  }
  return true;
}
}  // namespace tools

#endif  // TOOLS__FS_TOOLS_HPP
