// 检测预览：相机取帧 → 0526 YOLO（TensorRT）→ 画四点框/标签 → 浏览器 MJPEG / 窗口
//
// 两种模式同 camera_view（--serve 走 MJPEG，见 tools/mjpeg_server.hpp）
// 注意：叠加文字只能用 ASCII——Hershey 字体渲染不了中文；参数必须用 "=" 传值
// 用法：
//   ./detect_view --serve=8080 --scale=0.75    # 浏览器开 http://10.42.0.80:8080/
//   ./detect_view                              # 窗口模式
//   ./detect_view --conf=0.6 --nms=0.4         # 覆盖 yaml 里的 yolo 阈值
// 快捷键（窗口模式）：q / ESC 退出；s 存干净图；w 重做一次白平衡

#include <csignal>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "cli_common.hpp"
#include "io/galaxy/galaxy.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/mjpeg_server.hpp"

const cv::String keys =
  "{help h usage ? |      | 打印帮助 }"
  "{@config-path  | configs/standard3.yaml | 配置文件路径（galaxy: 相机 + yolo: 检测）}"
  "{exposure e    | -1   | 曝光时间 µs（>0 才生效）}"
  "{gain g        | -1   | 增益 dB（>=0 才生效）}"
  "{fps f         | -1   | 目标帧率 Hz（>0 才生效）}"
  "{roi           |      | ROI 尺寸 WxH，如 1024x768（不改则不填）}"
  "{decimation d  | 0    | 抽行抽列 1~2（>0 才生效）}"
  "{wb w          |      | 白平衡 Off/Once/Continuous（空 = 按 yaml）}"
  "{scale         | 1.0  | 显示缩放 0.1~1.0（画面小一圈、省 CPU）}"
  "{serve         | 0    | >0 时不开窗口，改在该端口提供 MJPEG 网页流}"
  "{conf c        | 0    | objectness 阈值（>0 覆盖 yaml）}"
  "{nms n         | 0    | NMS 阈值（>0 覆盖 yaml）}";

namespace
{

/// 滚动窗口分位数（耗时/帧率统计用）
struct Stat
{
  std::deque<double> win;

  void push(double v)
  {
    win.push_back(v);
    while (win.size() > 300) win.pop_front();
  }

  double pct(double p) const
  {
    if (win.empty()) return 0;
    std::vector<double> v(win.begin(), win.end());
    std::sort(v.begin(), v.end());
    return v[(std::size_t)(p * (v.size() - 1))];
  }
};

/// 画四点框 + 角点 + 标签（坐标为原图坐标，需按显示缩放换算）
void draw_armors(cv::Mat & img, const std::list<auto_aim::Armor> & armors, double scale)
{
  for (const auto & a : armors) {
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (const auto & p : a.points)
      poly.emplace_back((int)(p.x * scale), (int)(p.y * scale));

    cv::polylines(img, poly, true, {0, 255, 0}, 2);
    for (const auto & p : poly) cv::circle(img, p, 5, {0, 0, 255}, -1);

    const std::string label = cv::format(
      "%s %s %.2f %s", auto_aim::COLORS[a.color].c_str(), auto_aim::ARMOR_NAMES[a.name].c_str(),
      a.confidence, auto_aim::ARMOR_TYPES[a.type].c_str());
    cv::Point org = poly[0] + cv::Point(-40, -14);
    org.x = std::max(org.x, 2);
    org.y = std::max(org.y, 20);
    cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
  }
}

void draw_overlay(
  cv::Mat & img, double fps, double fps_avg, double cam_hz, double lat_ms, uint64_t n,
  std::size_t det_count, const auto_aim::YOLO::Timings & timings, const auto_aim::YoloConfig & cfg,
  const io::GalaxyConfig & cam)
{
  const cv::Rect box(0, 0, 900, 150);
  cv::Mat roi = img(box);
  cv::Mat black(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
  cv::addWeighted(roi, 0.35, black, 0.65, 0, roi);

  const auto green = cv::Scalar(0, 255, 0);
  const auto yellow = cv::Scalar(0, 255, 255);
  cv::putText(
    img, cv::format("FPS %.1f  det %zu", fps, det_count), {12, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.1,
    green, 3);
  cv::putText(
    img, cv::format("(2s avg %.1f  cam %.1fHz)", fps_avg, cam_hz), {300, 38},
    cv::FONT_HERSHEY_SIMPLEX, 0.65, yellow, 2);
  cv::putText(
    img, cv::format("pre %.2f  infer %.2f  post %.2f  total %.2f ms", timings.preprocess,
                    timings.inference, timings.postprocess, timings.total()),
    {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.75, yellow, 2);
  cv::putText(
    img,
    cv::format(
      "%dx%d  lat %.1fms  frame #%llu  conf %.2f nms %.2f", cam.roi_width, cam.roi_height, lat_ms,
      (unsigned long long)n, cfg.min_confidence, cfg.nms_threshold),
    {12, 113}, cv::FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2);

  std::string bottom =
    cam.exposure_us > 0 ? cv::format("exp %.0fus", cam.exposure_us) : "exp camera-default";
  if (cam.gain_db >= 0) bottom += cv::format(" gain %.1fdB", cam.gain_db);
  if (cam.decimation > 1) bottom += cv::format(" decim %d", cam.decimation);
  bottom += "   [q]uit [s]nap [w]WB";
  cv::putText(img, bottom, {12, 142}, cv::FONT_HERSHEY_SIMPLEX, 0.65, yellow, 2);
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
  if (!cli_tools::config_path_ok("detect_view", config_path)) return 1;
  auto camera_config = io::galaxy_config_from_yaml(config_path);
  if (!cli_tools::apply_camera_overrides(cli, camera_config, "detect_view")) return 1;

  auto yolo_config = auto_aim::yolo_config_from_yaml(config_path);
  double conf = cli.get<double>("conf");
  if (conf > 0) yolo_config.min_confidence = conf;
  double nms = cli.get<double>("nms");
  if (nms > 0) yolo_config.nms_threshold = nms;

  double scale = cli.get<double>("scale");
  if (scale <= 0 || scale > 1) scale = 1.0;
  int port = cli.get<int>("serve");
  const bool serve_mode = port > 0;

  std::unique_ptr<auto_aim::YOLO> yolo;
  try {
    yolo.reset(new auto_aim::YOLO(yolo_config));
  } catch (const std::exception & e) {
    tools::logger()->error("[detect_view] 载入检测模型失败: {}", e.what());
    return 1;
  }

  tools::Exiter exiter;
  io::Galaxy camera(camera_config);

  cv::Mat raw, bgr, disp;
  std::chrono::steady_clock::time_point ts;
  for (int i = 0; i < 5; i++) camera.read(raw, ts);  // 预热
  if (raw.empty()) {
    tools::logger()->error("[detect_view] 预热没拿到帧，退出");
    return 1;
  }
  yolo->detect(raw);  // 预热一次推理（首次含 CUDA 初始化，别让它污染统计）

  const std::string title = "rm_vision detect view";
  tools::MjpegStreamer streamer;
  std::atomic<bool> quit{false};
  std::thread server;
  if (serve_mode) {
    server = std::thread(tools::serve_mjpeg, port, std::ref(streamer), std::ref(quit), "detect_view");
    tools::logger()->info("[detect_view] MJPEG 模式（端口 {}），Ctrl+C 退出", port);
  } else {
    tools::logger()->info("[detect_view] 窗口模式；无显示环境时请改用 --serve=8080");
  }

  std::deque<std::chrono::steady_clock::time_point> win;
  Stat st_pre, st_infer, st_post, st_read, st_draw, st_fps;
  bool ever_visible = false;  // 窗口是否曾经可见过（区分启动竞态与真的被关掉）
  uint64_t n = 0;
  std::size_t det_count = 0;
  auto_aim::YOLO::Timings timings;
  auto t0 = std::chrono::steady_clock::now();
  auto t_last_log = t0, t_last_cam = t0, t_last_enc = t0;
  double fps = 0, fps_avg = 0, cam_hz = 0, lat_ms = 0, draw_ms = 0;

  while (!exiter.exit()) {
    auto t_before_read = std::chrono::steady_clock::now();
    camera.read(raw, ts);
    if (raw.empty()) continue;
    auto now = std::chrono::steady_clock::now();
    st_read.push(std::chrono::duration<double, std::milli>(now - t_before_read).count());

    // 延迟要在检测之前量（检测耗时 ~10ms，含进去就把画面延迟算多了）
    lat_ms = tools::delta_time(now, ts) * 1000.0;

    std::list<auto_aim::Armor> armors;
    try {
      armors = yolo->detect(raw);
      timings = yolo->last_timings();
    } catch (const std::exception & e) {
      tools::logger()->error("[detect_view] 检测失败: {}", e.what());
      break;
    }
    det_count = armors.size();
    st_pre.push(timings.preprocess);
    st_infer.push(timings.inference);
    st_post.push(timings.postprocess);

    n++;
    win.push_back(now);
    while (!win.empty() && tools::delta_time(now, win.front()) > 2.0) win.pop_front();
    double span = win.empty() ? 0 : tools::delta_time(now, win.front());
    fps = span > 0.3 ? (win.size() - 1) / span : 0.0;
    fps_avg = tools::delta_time(now, t0) > 0.3 ? n / tools::delta_time(now, t0) : 0.0;
    st_fps.push(fps);
    if (tools::delta_time(now, t_last_cam) >= 1.0) {  // 相机自报值每秒查一次
      cam_hz = camera.actual_framerate();
      t_last_cam = now;
    }

    // 网络模式下按 ~30fps 出图即可（编码很贵，不必每帧都做）
    bool need_draw = !serve_mode || tools::delta_time(now, t_last_enc) >= 0.033;
    if (need_draw) {
      auto t_draw0 = std::chrono::steady_clock::now();
      cv::cvtColor(raw, bgr, cv::COLOR_BayerBG2BGR);  // RGGB 布局 → BG 常量（见台账 §5）
      if (scale != 1.0)
        cv::resize(bgr, disp, {}, scale, scale, cv::INTER_AREA);
      else
        disp = bgr;
      draw_armors(disp, armors, scale);
      draw_overlay(disp, fps, fps_avg, cam_hz, lat_ms, n, det_count, timings, yolo_config, camera_config);

      if (serve_mode) {
        streamer.publish(disp);  // JPEG 编码在服务线程做（见 tools/mjpeg_server.hpp 注释）
        t_last_enc = now;
      } else {
        try {
          cv::imshow(title, disp);
          int k = cv::waitKey(1);
          if (k == 'q' || k == 27) {
            tools::logger()->info("[detect_view] 按键退出");
            break;
          }
          if (k == 's') cli_tools::snapshot(bgr, "detect_view");
          if (k == 'w') camera.white_balance_once();
          // 窗口可见性：GTK 下新窗口映射有延迟，首帧可能短暂不可见 → 只有"曾经可见过"才算被点 X 关掉
          if (cv::getWindowProperty(title, cv::WND_PROP_VISIBLE) >= 1) {
            ever_visible = true;
          } else if (ever_visible) {
            tools::logger()->info("[detect_view] 窗口已关闭");
            break;
          }
        } catch (const cv::Exception & e) {
          tools::logger()->error(
            "[detect_view] 打不开窗口（{}）——无显示环境请用 --serve=8080", e.what());
          break;
        }
      }
      draw_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t_draw0)
                  .count();
      st_draw.push(draw_ms);
    }

    if (tools::delta_time(now, t_last_log) >= 3.0) {
      tools::logger()->info(
        "[detect_view] fps={:.1f} 相机自报={:.1f}Hz 延迟={:.1f}ms 帧数={} 检出={} | "
        "读帧 p50={:.2f}/p99={:.2f} 预处理 p50={:.2f}/p99={:.2f} 推理 p50={:.2f}/p99={:.2f} "
        "解码 p50={:.2f}/p99={:.2f} 绘制 p50={:.2f}/p99={:.2f} ms",
        fps, cam_hz, lat_ms, n, det_count, st_read.pct(0.5), st_read.pct(0.99), st_pre.pct(0.5),
        st_pre.pct(0.99), st_infer.pct(0.5), st_infer.pct(0.99), st_post.pct(0.5),
        st_post.pct(0.99), st_draw.pct(0.5), st_draw.pct(0.99));
      t_last_log = now;
    }
  }

  // 收尾统计（耗时分解台账用）
  auto t_end = std::chrono::steady_clock::now();
  if (n > 0) {
    tools::logger()->info(
      "[detect_view] 汇总：运行 {:.1f}s 帧数 {} 平均fps={:.1f} | 读帧 p50={:.2f}/p99={:.2f} "
      "预处理 p50={:.2f}/p99={:.2f} 推理 p50={:.2f}/p99={:.2f} 解码 p50={:.2f}/p99={:.2f} "
      "绘制 p50={:.2f}/p99={:.2f} ms",
      tools::delta_time(t_end, t0), n, n / std::max(0.001, tools::delta_time(t_end, t0)),
      st_read.pct(0.5), st_read.pct(0.99), st_pre.pct(0.5), st_pre.pct(0.99), st_infer.pct(0.5),
      st_infer.pct(0.99), st_post.pct(0.5), st_post.pct(0.99), st_draw.pct(0.5),
      st_draw.pct(0.99));
  }

  quit = true;
  if (server.joinable()) server.join();
  tools::logger()->info("[detect_view] 退出");
  return 0;
}
