#include "galaxy.hpp"

#include <DxImageProc.h>
#include <GxIAPI.h>

#include <cstring>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace io
{

namespace
{
using GXH = GX_DEV_HANDLE;

bool check(GX_STATUS status, const std::string & what)
{
  if (status == GX_STATUS_SUCCESS) return true;

  char msg[256] = {0};
  size_t size = sizeof(msg);
  GXGetLastError(&status, msg, &size);
  tools::logger()->error("[Galaxy] {} 失败: {}", what, msg);
  return false;
}

bool implemented(GXH h, const char * name)
{
  GX_NODE_ACCESS_MODE mode;
  if (GXGetNodeAccessMode(h, name, &mode) != GX_STATUS_SUCCESS) return false;
  return mode != GX_NODE_ACCESS_MODE_NI && mode != GX_NODE_ACCESS_MODE_NA;
}

bool set_float(GXH h, const char * name, double value)
{
  if (!implemented(h, name)) {
    tools::logger()->warn("[Galaxy] 相机不支持特征 {}，忽略", name);
    return false;
  }
  if (GXSetFloatValue(h, name, value) != GX_STATUS_SUCCESS) {
    tools::logger()->warn("[Galaxy] 设置 {}={} 失败（超出范围？）", name, value);
    return false;
  }
  return true;
}

bool set_int(GXH h, const char * name, int64_t value)
{
  if (!implemented(h, name)) {
    tools::logger()->warn("[Galaxy] 相机不支持特征 {}，忽略", name);
    return false;
  }
  if (GXSetIntValue(h, name, value) != GX_STATUS_SUCCESS) {
    tools::logger()->warn("[Galaxy] 设置 {}={} 被相机拒绝（可能超当前可设范围）", name, (long long)value);
    return false;
  }
  return true;
}

bool set_enum(GXH h, const char * name, const char * value)
{
  if (!implemented(h, name)) {
    tools::logger()->warn("[Galaxy] 相机不支持特征 {}，忽略", name);
    return false;
  }
  if (GXSetEnumValueByString(h, name, value) != GX_STATUS_SUCCESS) {
    tools::logger()->warn("[Galaxy] 设置 {}={} 被相机拒绝", name, value);
    return false;
  }
  return true;
}

double get_float(GXH h, const char * name, double fallback = 0)
{
  GX_FLOAT_VALUE v;
  if (GXGetFloatValue(h, name, &v) != GX_STATUS_SUCCESS) return fallback;
  return v.dCurValue;
}

std::string get_string(GXH h, const char * name)
{
  GX_STRING_VALUE v;
  if (GXGetStringValue(h, name, &v) != GX_STATUS_SUCCESS) return "";
  return std::string(v.strCurValue);
}
}  // namespace

GalaxyConfig galaxy_config_from_yaml(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  if (!yaml["galaxy"]) throw std::runtime_error("yaml 缺少 galaxy: 相机配置块");

  auto g = yaml["galaxy"];
  GalaxyConfig config;
  config.sn = tools::read_or<std::string>(g, "sn", "");
  config.index = tools::read_or<int>(g, "index", 1);
  config.exposure_us = tools::read_or<double>(g, "exposure_us", -1);
  config.gain_db = tools::read_or<double>(g, "gain_db", -1);
  config.framerate = tools::read_or<double>(g, "framerate", -1);
  config.pixel_format = tools::read_or<std::string>(g, "pixel_format", "");
  config.roi_width = tools::read_or<int>(g, "roi_width", -1);
  config.roi_height = tools::read_or<int>(g, "roi_height", -1);
  config.roi_x = tools::read_or<int>(g, "roi_x", -1);
  config.roi_y = tools::read_or<int>(g, "roi_y", -1);
  config.decimation = tools::read_or<int>(g, "decimation", -1);
  return config;
}

Galaxy::Galaxy(const GalaxyConfig & config) : config_(config)
{
  if (!check(GXInitLib(), "GXInitLib")) throw std::runtime_error("Galaxy SDK 初始化失败");

  try_open();  // 首次失败不致命：守护线程会持续重试
  daemon_thread_ = std::thread(&Galaxy::daemon_loop, this);
}

Galaxy::Galaxy(const std::string & config_path) : Galaxy(galaxy_config_from_yaml(config_path)) {}

Galaxy::~Galaxy()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  stop_capture();
  queue_.close();
  GXCloseLib();
  tools::logger()->info("[Galaxy] 已关闭");
}

void Galaxy::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);
  img = data.img;
  timestamp = data.timestamp;
}

double Galaxy::actual_framerate() const
{
  if (handle_ == nullptr) return 0;
  return get_float((GXH)handle_, "CurrentAcquisitionFrameRate", 0);
}

void Galaxy::try_open()
{
  try {
    open();
  } catch (const std::exception & e) {
    tools::logger()->warn("[Galaxy] 打开相机失败: {}（守护线程将重试）", e.what());
  }
}

void Galaxy::open()
{
  uint32_t num = 0;
  if (!check(GXUpdateAllDeviceList(&num, 1000), "GXUpdateAllDeviceList"))
    throw std::runtime_error("枚举设备失败");
  if (num == 0) throw std::runtime_error("未发现相机（检查 USB 连接）");

  GXH h = nullptr;
  if (!config_.sn.empty()) {
    char sn[64] = {0};
    std::strncpy(sn, config_.sn.c_str(), sizeof(sn) - 1);
    GX_OPEN_PARAM param;
    param.pszContent = sn;
    param.openMode = GX_OPEN_SN;
    param.accessMode = GX_ACCESS_EXCLUSIVE;
    if (!check(GXOpenDevice(&param, &h), "GXOpenDevice(SN=" + config_.sn + ")"))
      throw std::runtime_error("按序列号打开失败");
  } else {
    if (!check(GXOpenDeviceByIndex((uint32_t)config_.index, &h), "GXOpenDeviceByIndex"))
      throw std::runtime_error("按序号打开失败");
  }
  handle_ = h;

  tools::logger()->info(
    "[Galaxy] 已打开: {} {} (SN {} / 固件 {})", get_string(h, "DeviceVendorName"),
    get_string(h, "DeviceModelName"), get_string(h, "DeviceSerialNumber"),
    get_string(h, "DeviceFirmwareVersion"));

  apply_config();

  // 缓冲区 2：搭配"丢旧留新"的队列，尽量压低延迟
  if (!check(GXSetAcqusitionBufferNumber(h, 2), "GXSetAcqusitionBufferNumber"))
    throw std::runtime_error("设置采集缓冲数失败");
  if (!check(GXStreamOn(h), "GXStreamOn")) throw std::runtime_error("开流失败");

  capture_quit_ = false;
  capture_thread_ = std::thread(&Galaxy::capture_loop, this);
  ok_ = true;
  tools::logger()->info("[Galaxy] 取流已启动");
}

void Galaxy::apply_config()
{
  auto h = (GXH)handle_;

  set_enum(h, "AcquisitionMode", "Continuous");
  set_enum(h, "TriggerMode", "Off");
  set_enum(h, "BalanceWhiteAuto", "Off");

  if (!config_.pixel_format.empty()) set_enum(h, "PixelFormat", config_.pixel_format.c_str());

  // 抽行先归 1：实测相机遗留的抽行≠1 会缩窄 Width/Height 可设范围，让后面的 ROI 设置被拒
  if (config_.decimation >= 0) {
    set_int(h, "DecimationHorizontal", 1);
    set_int(h, "DecimationVertical", 1);
  }

  // ROI：先改尺寸、后挪偏移（偏移上限受当前尺寸约束）
  if (config_.roi_width >= 0) set_int(h, "Width", config_.roi_width);
  if (config_.roi_height >= 0) set_int(h, "Height", config_.roi_height);
  if (config_.roi_x >= 0) set_int(h, "OffsetX", config_.roi_x);
  if (config_.roi_y >= 0) set_int(h, "OffsetY", config_.roi_y);

  // 抽行/抽列：最后设目标值（该设置会按比例缩放 Width/Height 节点）
  if (config_.decimation >= 0) {
    set_int(h, "DecimationHorizontal", config_.decimation);
    set_int(h, "DecimationVertical", config_.decimation);
  }

  // 曝光/增益：先关自动，再设值
  if (config_.exposure_us > 0) {
    set_enum(h, "ExposureAuto", "Off");
    set_float(h, "ExposureTime", config_.exposure_us);
  }
  if (config_.gain_db >= 0) {
    set_enum(h, "GainAuto", "Off");
    set_float(h, "Gain", config_.gain_db);
  }

  // 帧率：先开帧率控制模式（若支持），再设值
  if (config_.framerate > 0) {
    set_enum(h, "AcquisitionFrameRateMode", "On");
    set_float(h, "AcquisitionFrameRate", config_.framerate);
  }

  // 回报关键实际值（读失败时留 0，不至于打印垃圾数）
  GX_FLOAT_VALUE exp{}, gain_v{}, fps{};
  GXGetFloatValue(h, "ExposureTime", &exp);
  GXGetFloatValue(h, "Gain", &gain_v);
  GXGetFloatValue(h, "AcquisitionFrameRate", &fps);
  GX_INT_VALUE w{}, hgt{};
  GXGetIntValue(h, "Width", &w);
  GXGetIntValue(h, "Height", &hgt);
  GX_INT_VALUE offx{}, offy{};
  GXGetIntValue(h, "OffsetX", &offx);
  GXGetIntValue(h, "OffsetY", &offy);
  GX_INT_VALUE dec_h{}, dec_v{};
  GXGetIntValue(h, "DecimationHorizontal", &dec_h);
  GXGetIntValue(h, "DecimationVertical", &dec_v);

  tools::logger()->info(
    "[Galaxy] 实际参数: {}({}x{}+{}+{}) 抽行 {}/{} 曝光 {:.0f}µs[{:.0f},{:.0f}] 增益 "
    "{:.1f}dB[{:.1f},{:.1f}] 帧率设定 {:.1f}Hz",
    get_string(h, "PixelFormat"), w.nCurValue, hgt.nCurValue, offx.nCurValue, offy.nCurValue,
    dec_h.nCurValue, dec_v.nCurValue, exp.dCurValue, exp.dMin, exp.dMax, gain_v.dCurValue,
    gain_v.dMin, gain_v.dMax, fps.dCurValue);
}

void Galaxy::capture_loop()
{
  auto h = (GXH)handle_;
  tools::logger()->debug("[Galaxy] 捕获线程启动");

  while (!capture_quit_) {
    PGX_FRAME_BUFFER frame = nullptr;
    GX_STATUS status = GXDQBuf(h, &frame, 1000);

    if (status == GX_STATUS_TIMEOUT) continue;
    if (status != GX_STATUS_SUCCESS) {
      char msg[256] = {0};
      size_t size = sizeof(msg);
      GXGetLastError(&status, msg, &size);
      tools::logger()->warn("[Galaxy] 取帧失败: {}", msg);
      ok_ = false;
      break;
    }

    if (frame->nStatus == GX_FRAME_STATUS_SUCCESS) {
      const size_t raw_bytes = (size_t)frame->nWidth * frame->nHeight;
      if ((size_t)frame->nImgSize != raw_bytes) {
        // 非 8bit 像素格式（如 BayerRG10）：当前驱动只按 BayerRG8 处理，丢弃并提醒
        static bool warned = false;
        if (!warned) {
          warned = true;
          tools::logger()->warn(
            "[Galaxy] 像素格式不是 8bit（nImgSize={}, 预期 {}），请把 pixel_format 设为 BayerRG8",
            (size_t)frame->nImgSize, raw_bytes);
        }
      } else {
        cv::Mat raw(frame->nHeight, frame->nWidth, CV_8UC1, frame->pImgBuf);
        CameraData data;
        data.img = raw.clone();  // GXQBuf 后 SDK 会回收该 buffer，必须拷贝
        // TODO(A3): 同时记录 frame->nTimestamp（相机时钟）并与主机时间对齐
        data.timestamp = std::chrono::steady_clock::now();
        queue_.push(data);
      }
    } else {
      tools::logger()->warn("[Galaxy] 异常帧, status={}", (int)frame->nStatus);
    }

    if (GXQBuf(h, frame) != GX_STATUS_SUCCESS) {
      tools::logger()->warn("[Galaxy] 归还缓冲失败");
      ok_ = false;
      break;
    }
  }

  tools::logger()->debug("[Galaxy] 捕获线程退出");
}

void Galaxy::daemon_loop()
{
  int fail_count = 0;
  while (!daemon_quit_) {
    std::this_thread::sleep_for(100ms);

    if (ok_) {
      fail_count = 0;
      continue;
    }

    // 每 5s 报一次，避免刷屏
    if (fail_count++ % 50 == 0) tools::logger()->warn("[Galaxy] 取流异常，尝试重开相机…");

    stop_capture();
    try_open();
  }
}

void Galaxy::stop_capture()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  if (handle_ != nullptr) {
    GXStreamOff((GXH)handle_);
    close();
  }
  ok_ = false;
}

void Galaxy::close()
{
  if (handle_ == nullptr) return;
  GXCloseDevice((GXH)handle_);
  handle_ = nullptr;
}
}  // namespace io
