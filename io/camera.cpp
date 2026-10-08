#include "camera.hpp"

#include <stdexcept>

#include "galaxy/galaxy.hpp"
#include "tools/yaml.hpp"

namespace io
{
Camera::Camera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto camera_name = tools::read<std::string>(yaml, "camera_name");

  if (camera_name == "galaxy") {
    camera_ = std::make_unique<Galaxy>(config_path);
  }

  else {
    throw std::runtime_error("未知 camera_name: " + camera_name);
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  camera_->read(img, timestamp);
}
}  // namespace io
