#pragma once

#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>

#include "gimbal.hpp"

namespace io {

class MockGimbal {
public:
  MockGimbal(const std::string & config_path) {
    // 初始化模拟数据
    state_.yaw = 0.0f;
    state_.yaw_vel = 0.0f;
    state_.pitch = 0.0f;
    state_.pitch_vel = 0.0f;
    state_.bullet_speed = 15.0f;  // 默认子弹速度
    state_.bullet_count = 0;
    
    // 启动模拟线程
    thread_ = std::thread(&MockGimbal::simulate_thread, this);
  }

  ~MockGimbal() {
    quit_ = true;
    if (thread_.joinable()) thread_.join();
  }

  GimbalMode mode() const {
    return GimbalMode::AUTO_AIM;  // 默认自瞄模式
  }

  GimbalState state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

  std::string str(GimbalMode mode) const {
    switch (mode) {
      case GimbalMode::IDLE: return "IDLE";
      case GimbalMode::AUTO_AIM: return "AUTO_AIM";
      case GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
      case GimbalMode::BIG_BUFF: return "BIG_BUFF";
      default: return "INVALID";
    }
  }

  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t) {
    // 返回单位四元数（无旋转）
    return Eigen::Quaterniond::Identity();
  }

  void send(bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, 
            float pitch, float pitch_vel, float pitch_acc) {
    // 模拟发送控制指令
    if (fire) {
      std::lock_guard<std::mutex> lock(mutex_);
      state_.bullet_count++;
    }
  }

  void send(VisionToGimbal vision_to_gimbal) {
    // 模拟发送控制指令
    if (vision_to_gimbal.mode == 2) {  // 开火模式
      std::lock_guard<std::mutex> lock(mutex_);
      state_.bullet_count++;
    }
  }

private:
  void simulate_thread() {
    while (!quit_) {
      // 模拟云台状态更新
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;
  GimbalState state_;
};

} // namespace io