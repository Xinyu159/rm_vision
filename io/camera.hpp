#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

namespace io
{
/// 相机驱动抽象（语义对齐 sp_vision_25 的 io::CameraBase）
/// read() 阻塞，返回"当前最新帧"及其采样时刻（steady_clock）
class CameraBase
{
public:
  virtual ~CameraBase() = default;
  virtual void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) = 0;
};

/// 工厂：按 yaml 的 camera_name 选驱动（当前只有 galaxy，后续可加 usbcamera 等）
class Camera
{
public:
  Camera(const std::string & config_path);
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

private:
  std::unique_ptr<CameraBase> camera_;
};
}  // namespace io

#endif  // IO__CAMERA_HPP
