// 相机实时预览：取流 + 画面 + 帧率/延迟叠加显示
//
// 两种模式：
//   窗口模式（默认）  ：在带显示器的机器上直接开窗口（Jetson 本机：DISPLAY=:1）
//   网络模式（--serve）：内置 MJPEG HTTP 服务（tools/mjpeg_server.hpp），浏览器打开
//                        http://<Jetson IP>:<端口>/ 即可看（无显示器 / SSH 场景用）
//
// 注意：叠加文字只能用 ASCII——cv::putText 的 Hershey 字体不支持中文，中文会渲染成"??"
// 注意：参数必须用 "=" 传值（--exposure=4000），空格分隔会把数值当成配置文件路径
// 用法：
//   ./camera_view                          # 默认参数开窗口
//   ./camera_view --exposure=4000          # 曝光 4ms
//   ./camera_view --scale=0.75             # 显示缩放（GTK 每帧拷贝，全幅约 30fps 上限）
//   ./camera_view --wb=Once                # 进画面后做一次白平衡（默认按 yaml）
//   ./camera_view --serve=8080             # 不开窗口，浏览器看流
// 快捷键（窗口模式）：q / ESC 退出；s 存干净图；w 重做一次白平衡

#include <csignal>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "cli_common.hpp"
#include "io/galaxy/galaxy.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/mjpeg_server.hpp"

const cv::String keys =
  "{help h usage ? |      | 打印帮助 }"
  "{@config-path  | configs/standard3.yaml | 配置文件路径（galaxy: 块作为默认参数）}"
  "{exposure e    | -1   | 曝光时间 µs（>0 才生效）}"
  "{gain g        | -1   | 增益 dB（>=0 才生效）}"
  "{fps f         | -1   | 目标帧率 Hz（>0 才生效）}"
  "{roi           |      | ROI 尺寸 WxH，如 1024x768（不改则不填）}"
  "{decimation d  | 0    | 抽行抽列 1~2（>0 才生效）}"
  "{wb w          |      | 白平衡 Off/Once/Continuous（空 = 按 yaml）}"
  "{scale         | 1.0  | 显示缩放 0.1~1.0（画面小一圈、省 CPU）}"
  "{serve         | 0    | >0 时不开窗口，改在该端口提供 MJPEG 网页流}";

namespace
{

void draw_overlay(
  cv::Mat & img, double fps, double fps_avg, double cam_hz, double lat_ms, uint64_t n, int raw_w,
  int raw_h, const io::GalaxyConfig & cfg)
{
  const cv::Rect box(0, 0, 640, 122);
  cv::Mat roi = img(box);
  cv::Mat black(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
  cv::addWeighted(roi, 0.35, black, 0.65, 0, roi);

  const auto green = cv::Scalar(0, 255, 0);
  const auto yellow = cv::Scalar(0, 255, 255);
  cv::putText(img, cv::format("FPS %.1f", fps), {12, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.2, green, 3);
  cv::putText(
    img, cv::format("(2s avg %.1f  cam %.1fHz)", fps_avg, cam_hz), {250, 38},
    cv::FONT_HERSHEY_SIMPLEX, 0.65, yellow, 2);
  cv::putText(
    img, cv::format("%dx%d  lat %.1fms  frame #%llu", raw_w, raw_h, lat_ms, (unsigned long long)n),
    {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.75, yellow, 2);

  std::string bottom =
    cfg.exposure_us > 0 ? cv::format("exp %.0fus", cfg.exposure_us) : "exp camera-default";
  if (cfg.gain_db >= 0) bottom += cv::format(" gain %.1fdB", cfg.gain_db);
  if (cfg.decimation > 1) bottom += cv::format(" decim %d", cfg.decimation);
  bottom += "   [q]uit [s]nap [w]WB";
  cv::putText(img, bottom, {12, 110}, cv::FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2);
}

}  // namespace

int main(int argc, char ** argv)
{
  std::signal(SIGPIPE, SIG_IGN);  // 浏览器断开时别被 SIGPIPE 打死

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<cv::String>("@config-path");
  if (!cli_tools::config_path_ok("camera_view", config_path)) return 1;
  auto config = io::galaxy_config_from_yaml(config_path);
  if (!cli_tools::apply_camera_overrides(cli, config, "camera_view")) return 1;

  double scale = cli.get<double>("scale");
  if (scale <= 0 || scale > 1) scale = 1.0;
  int port = cli.get<int>("serve");
  const bool serve_mode = port > 0;

  tools::Exiter exiter;
  io::Galaxy camera(config);

  cv::Mat raw, bgr, disp;
  std::chrono::steady_clock::time_point ts;
  for (int i = 0; i < 5; i++) camera.read(raw, ts);  // 预热
  if (raw.empty()) {
    tools::logger()->error("[camera_view] 预热没拿到帧，退出");
    return 1;
  }

  const std::string title = "rm_vision camera view";
  tools::MjpegStreamer streamer;
  std::atomic<bool> quit{false};
  std::thread server;
  if (serve_mode) {
    server = std::thread(tools::serve_mjpeg, port, std::ref(streamer), std::ref(quit), "camera_view");
    tools::logger()->info("[camera_view] MJPEG 模式（端口 {}），Ctrl+C 退出", port);
  } else {
    tools::logger()->info("[camera_view] 窗口模式；无显示环境时请改用 --serve=8080");
  }

  std::deque<std::chrono::steady_clock::time_point> win;
  bool ever_visible = false;  // 窗口是否曾经可见过（区分启动竞态与真的被关掉）
  uint64_t n = 0;
  auto t0 = std::chrono::steady_clock::now();
  auto t_last_log = t0, t_last_cam = t0, t_last_enc = t0;
  double fps = 0, fps_avg = 0, cam_hz = 0, lat_ms = 0;

  while (!exiter.exit()) {
    camera.read(raw, ts);
    if (raw.empty()) continue;
    auto now = std::chrono::steady_clock::now();

    n++;
    lat_ms = tools::delta_time(now, ts) * 1000.0;
    win.push_back(now);
    while (!win.empty() && tools::delta_time(now, win.front()) > 2.0) win.pop_front();
    double span = win.empty() ? 0 : tools::delta_time(now, win.front());
    fps = span > 0.3 ? (win.size() - 1) / span : 0.0;
    fps_avg = tools::delta_time(now, t0) > 0.3 ? n / tools::delta_time(now, t0) : 0.0;
    if (tools::delta_time(now, t_last_cam) >= 1.0) {  // 相机自报值每秒查一次
      cam_hz = camera.actual_framerate();
      t_last_cam = now;
    }

    // 网络模式下按 ~30fps 出图即可（编码很贵，不必每帧都做）
    bool need_draw = !serve_mode || tools::delta_time(now, t_last_enc) >= 0.033;
    if (need_draw) {
      cv::cvtColor(raw, bgr, cv::COLOR_BayerBG2BGR);  // RGGB 布局 → BG 常量（见台账 §5）
      if (scale != 1.0)
        cv::resize(bgr, disp, {}, scale, scale, cv::INTER_AREA);
      else
        disp = bgr;
      draw_overlay(disp, fps, fps_avg, cam_hz, lat_ms, n, raw.cols, raw.rows, config);

      if (serve_mode) {
        streamer.publish(disp);  // JPEG 编码在服务线程做（见 tools/mjpeg_server.hpp 注释）
        t_last_enc = now;
      } else {
        try {
          cv::imshow(title, disp);
          int k = cv::waitKey(1);
          if (k == 'q' || k == 27) {
            tools::logger()->info("[camera_view] 按键退出");
            break;
          }
          if (k == 's') cli_tools::snapshot(bgr, "camera_view");
          if (k == 'w') camera.white_balance_once();
          // 窗口可见性：GTK 下新窗口映射有延迟，首帧可能短暂不可见 → 只有"曾经可见过"才算被点 X 关掉
          if (cv::getWindowProperty(title, cv::WND_PROP_VISIBLE) >= 1) {
            ever_visible = true;
          } else if (ever_visible) {
            tools::logger()->info("[camera_view] 窗口已关闭");
            break;
          }
        } catch (const cv::Exception & e) {
          tools::logger()->error(
            "[camera_view] 打不开窗口（{}）——无显示环境请用 --serve=8080", e.what());
          break;
        }
      }
    }

    if (tools::delta_time(now, t_last_log) >= 3.0) {
      tools::logger()->info(
        "[camera_view] fps={:.1f} 2s均值={:.1f} 相机自报={:.1f}Hz 延迟={:.1f}ms 帧数={}", fps, fps_avg,
        cam_hz, lat_ms, n);
      t_last_log = now;
    }
  }

  quit = true;
  if (server.joinable()) server.join();
  tools::logger()->info("[camera_view] 退出");
  return 0;
}
