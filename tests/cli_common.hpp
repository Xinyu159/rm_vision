// camera_test / camera_view 共用的命令行处理（两处维护同一套覆盖逻辑容易漏改，抽出来）
#ifndef TESTS__CLI_COMMON_HPP
#define TESTS__CLI_COMMON_HPP

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>

#include "io/galaxy/galaxy.hpp"
#include "tools/logger.hpp"

namespace cli_tools
{

/// 位置参数防呆：本工程参数必须用 "=" 传值，漏写时数值会顶掉默认的 config-path
inline bool config_path_ok(const std::string & tag, const std::string & config_path)
{
  if (
    config_path.find('/') != std::string::npos || config_path.find(".yaml") != std::string::npos)
    return true;

  tools::logger()->error(
    "[{}] 位置参数 \"{}\" 不像配置文件路径。注意：本程序参数必须用 = 传值（如 --exposure=4000），"
    "写 \"--exposure 4000\" 会把 4000 当成配置文件",
    tag, config_path);
  return false;
}

/// 命令行覆盖相机配置（原则：给了才覆盖）。
/// 前提：解析器已声明 exposure/gain/fps/roi/decimation/wb 这几个键
inline bool apply_camera_overrides(
  const cv::CommandLineParser & cli, io::GalaxyConfig & config, const std::string & tag)
{
  double exposure = cli.get<double>("exposure");
  if (exposure > 0) config.exposure_us = exposure;

  double gain = cli.get<double>("gain");
  if (gain >= 0) config.gain_db = gain;

  double fps = cli.get<double>("fps");
  if (fps > 0) config.framerate = fps;

  int decimation = cli.get<int>("decimation");
  if (decimation > 0) config.decimation = decimation;

  cv::String roi = cli.get<cv::String>("roi");
  if (!roi.empty()) {
    int w = 0, h = 0;
    if (std::sscanf(roi.c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) {
      tools::logger()->error("[{}] roi 格式应为 WxH，如 1024x768", tag);
      return false;
    }
    config.roi_width = w;
    config.roi_height = h;
  }

  cv::String wb = cli.get<cv::String>("wb");
  if (!wb.empty()) config.balance_white_auto = wb;

  return true;
}
}  // namespace cli_tools

#endif  // TESTS__CLI_COMMON_HPP
