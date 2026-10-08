// 步兵 3V3 主程序（骨架）
//
// 现在的形态：主线程只做"取流 + 打印帧率"，用来验证相机链路。
// 按 docs/sp-vision-25-structure-and-threads.md 的图纸，后续在本文件按顺序长出：
//   阶段2  detect_thread  ：img → 缩放 640 → 异步推理 → 检测结果入队
//   阶段3  tracker.track  ：检测结果 + 时间戳 → 目标状态
//   阶段4  CommandGener   ：500Hz 独立线程消费目标状态 → aimer/shooter → cboard.send
// 主线程保留"模式机"角色（cboard.mode 切换功能组）。

#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const cv::String keys =
  "{help h usage ? |      | 打印帮助 }"
  "{@config-path  | configs/standard3.yaml | 配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const std::string config_path = cli.get<cv::String>("@config-path");

  tools::Exiter exiter;
  tools::logger()->info("[standard] 配置: {}", config_path);

  io::Camera camera(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto t_last = std::chrono::steady_clock::now();
  int frames = 0;

  while (!exiter.exit()) {
    camera.read(img, timestamp);

    // TODO(阶段2): detect_thread —— img 缩放 640 → 异步推理 → 检测结果入队
    // TODO(阶段3): tracker.track(detections, timestamp) → 目标状态
    // TODO(阶段4): CommandGener 线程消费目标状态（图像-姿态时间对齐）→ 发弹道

    frames++;
    auto now = std::chrono::steady_clock::now();
    double dt = tools::delta_time(now, t_last);
    if (dt > 2.0) {
      tools::logger()->info("[standard] 相机 {:.1f} fps, {}x{}", frames / dt, img.cols, img.rows);
      frames = 0;
      t_last = now;
    }
  }

  tools::logger()->info("[standard] 退出");
  return 0;
}
