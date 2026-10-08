#ifndef TOOLS__YAML_HPP
#define TOOLS__YAML_HPP

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

namespace tools
{
inline YAML::Node load(const std::string & path)
{
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("读取 yaml 失败: " + path + " → " + e.what());
  }
}

template <typename T>
T read(const YAML::Node & yaml, const std::string & key)
{
  if (!yaml[key]) throw std::runtime_error("yaml 缺少键: " + key);
  try {
    return yaml[key].as<T>();
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("yaml 键类型错误: " + key + " → " + e.what());
  }
}

/// 可选读取：键不存在时返回 fallback，不做类型校验之外的报错
template <typename T>
T read_or(const YAML::Node & yaml, const std::string & key, const T & fallback)
{
  if (!yaml[key]) return fallback;
  return yaml[key].as<T>();
}
}  // namespace tools

#endif  // TOOLS__YAML_HPP
