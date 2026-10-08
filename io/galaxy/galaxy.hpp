#ifndef IO__GALAXY_HPP
#define IO__GALAXY_HPP

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "io/camera.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{

/// 大恒 Galaxy 相机可调参数。负值 / 空串 = 该项不改动（用相机当前值）
///
/// 注意：相机设置会跨开关机保留，工程里应当把关心的项全部写死，
/// 否则会悄悄沿用上一次遗留的状态（实测踩过：ROI 640x512 残留）。
struct GalaxyConfig
{
  std::string sn;   // 序列号（空则按 index 打开）
  int index = 1;    // 设备序号（从 1 开始）
  double exposure_us = -1;
  double gain_db = -1;
  double framerate = -1;
  std::string pixel_format;                                      // 如 "BayerRG8"
  int roi_width = -1, roi_height = -1, roi_x = -1, roi_y = -1;   // ROI（-1 = 不改）
  int decimation = -1;                                           // 抽行/抽列 1~2（-1 = 不改）
};

/// 从 yaml 的 galaxy: 块读参数
GalaxyConfig galaxy_config_from_yaml(const std::string & config_path);

/// 大恒 MER 系列驱动（Galaxy SDK / GxIAPI）
///
/// 线程模型（对齐 sp_vision_25 的相机驱动）：
///   - 捕获线程：GXDQBuf 连续取流 → 原始 Bayer 帧送入容量 1 的队列（丢旧留新，永远最新帧）
///   - 守护线程：100ms 周期健康检查，取流异常自动关闭并重开相机
///   - read()：阻塞取队列里的最新帧
class Galaxy : public CameraBase
{
public:
  explicit Galaxy(const GalaxyConfig & config);
  explicit Galaxy(const std::string & config_path);
  ~Galaxy() override;

  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;

  /// 相机自报的实际帧率（需在取流过程中调用）
  double actual_framerate() const;

private:
  struct CameraData
  {
    cv::Mat img;  // 原始 BayerRG8（CV_8UC1）；转换位置另行决定（台账 A2）。
                  // 转 BGR 注意 OpenCV 命名反直觉：RGGB 布局要用 COLOR_BayerBG2BGR（台账 §5 实测）
    std::chrono::steady_clock::time_point timestamp;
  };

  GalaxyConfig config_;
  void * handle_ = nullptr;  // GX_DEV_HANDLE（头文件里不暴露 SDK 类型）
  std::atomic<bool> ok_{false};
  std::thread capture_thread_, daemon_thread_;
  std::atomic<bool> capture_quit_{false}, daemon_quit_{false};

  // 容量 1 + 丢旧留新：消费慢时新帧覆盖旧帧，read() 永远拿最新
  tools::ThreadSafeQueue<CameraData, true> queue_{1};

  void try_open();
  void open();
  void apply_config();
  void capture_loop();
  void daemon_loop();
  void stop_capture();
  void close();
};
}  // namespace io

#endif  // IO__GALAXY_HPP
