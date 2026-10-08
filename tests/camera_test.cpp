// 相机取流 + 帧率实测工具（台账 A1 收口用）
//
// 注意：OpenCV 的参数解析器必须用 "=" 传值，写 -e 4000 会把 4000 当成位置参数！
// 用法示例：
//   ./camera_test                                   # 用 yaml 默认参数测 3 秒
//   ./camera_test --exposure=4000                   # 曝光 4ms（别名 -e=4000）
//   ./camera_test --roi=1024x768 --decimation=2     # ROI + 抽行抽列
//   ./camera_test --seconds=5 --save=frame.png      # 存一帧（Bayer→BGR）
//
// 结束时输出一行机读结果，便于 bash 扫参数：
//   RESULT exposure_us=... roi=WxH decimation=... fps=... cam_fps=... frames=...

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "io/galaxy/galaxy.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const cv::String keys =
  "{help h usage ? |      | 打印帮助 }"
  "{@config-path  | configs/standard3.yaml | 配置文件路径（galaxy: 块作为默认参数）}"
  "{exposure e    | -1   | 曝光时间 µs（>0 才生效）}"
  "{gain g        | -1   | 增益 dB（>=0 才生效）}"
  "{fps f         | -1   | 目标帧率 Hz（>0 才生效）}"
  "{roi           |      | ROI 尺寸 WxH，如 1024x768（不改则不填）}"
  "{decimation d  | 0    | 抽行抽列 1~2（>0 才生效）}"
  "{seconds s     | 3    | 测量时长（秒）}"
  "{save          |      | 预热后存一帧到该路径（Bayer→BGR 的 png）}";

int main(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<cv::String>("@config-path");
  if (config_path.find('/') == std::string::npos && config_path.find(".yaml") == std::string::npos) {
    tools::logger()->error(
      "[camera_test] 位置参数 \"{}\" 不像配置文件路径。注意：本程序参数必须用 = 传值（如 "
      "--exposure=4000），写 \"--exposure 4000\" 会把 4000 当成配置文件",
      config_path);
    return 1;
  }
  auto config = io::galaxy_config_from_yaml(config_path);

  // 命令行覆盖 yaml
  double exposure = cli.get<double>("exposure");
  if (exposure > 0) config.exposure_us = exposure;
  double gain = cli.get<double>("gain");
  if (gain >= 0) config.gain_db = gain;
  double fps_target = cli.get<double>("fps");
  if (fps_target > 0) config.framerate = fps_target;
  int decimation = cli.get<int>("decimation");
  if (decimation > 0) config.decimation = decimation;
  cv::String roi = cli.get<cv::String>("roi");
  if (!roi.empty()) {
    int w = 0, h = 0;
    if (std::sscanf(roi.c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) {
      tools::logger()->error("[camera_test] roi 格式应为 WxH，如 1024x768");
      return 1;
    }
    config.roi_width = w;
    config.roi_height = h;
  }
  double seconds = cli.get<double>("seconds");

  tools::Exiter exiter;
  io::Galaxy camera(config);

  // 预热：丢掉前 10 帧，等流稳定（注意：这项会在测低帧率配置时拉长启动时间）
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  for (int i = 0; i < 10; i++) camera.read(img, timestamp);
  tools::logger()->info("[camera_test] 预热完成，实际出图 {}x{}", img.cols, img.rows);

  cv::String save = cli.get<cv::String>("save");
  if (!save.empty() && !img.empty()) {
    cv::Mat bgr;
    if (img.channels() == 1)
      cv::cvtColor(img, bgr, cv::COLOR_BayerRG2BGR);
    else
      bgr = img;
    cv::imwrite(save, bgr);
    tools::logger()->info("[camera_test] 已存帧: {}", (std::string)save);
  }

  // 看门狗：连续 3 秒没有新帧就判死退出，避免 read() 永久阻塞
  std::atomic<int> frame_count{0};
  std::atomic<bool> done{false};
  std::thread watchdog([&] {
    int last = -1, stuck = 0;
    while (!done) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      int now = frame_count.load();
      if (now == last && !done) {
        if (++stuck >= 3) {
          tools::logger()->error("[camera_test] 连续 3 秒无新帧，判定取流中断，退出");
          std::exit(1);
        }
      } else {
        stuck = 0;
      }
      last = now;
    }
  });

  tools::logger()->info("[camera_test] 开始测量 {:.1f}s ...", seconds);

  std::vector<double> intervals_ms;
  intervals_ms.reserve(4096);
  int64_t frames = 0;
  auto t_start = std::chrono::steady_clock::now();
  auto t_prev = t_start;

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    auto now = std::chrono::steady_clock::now();
    if (frames > 0) intervals_ms.push_back(tools::delta_time(now, t_prev) * 1000.0);
    t_prev = now;
    frames++;
    frame_count.store((int)frames);

    if (tools::delta_time(now, t_start) >= seconds) break;
  }

  done = true;
  watchdog.join();

  double elapsed = tools::delta_time(t_prev, t_start);
  double fps = frames / elapsed;
  double cam_fps = camera.actual_framerate();

  tools::logger()->info("[camera_test] 帧数 {} / {:.3f}s → {:.2f} fps", frames, elapsed, fps);

  if (!intervals_ms.empty()) {
    auto v = intervals_ms;
    std::sort(v.begin(), v.end());
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double avg = sum / v.size();
    double p50 = v[v.size() / 2];
    double p99 = v[std::min(v.size() - 1, (size_t)(v.size() * 0.99))];
    tools::logger()->info(
      "[camera_test] 帧间隔 ms: 最小 {:.3f}  p50 {:.3f}  平均 {:.3f}  p99 {:.3f}  最大 {:.3f}", v.front(),
      p50, avg, p99, v.back());
  }
  tools::logger()->info("[camera_test] 相机自报 CurrentAcquisitionFrameRate = {:.2f} Hz", cam_fps);

  // 机读结果行（bash 扫参数用）
  std::printf(
    "RESULT exposure_us=%.0f roi=%dx%d decimation=%d fps=%.2f cam_fps=%.2f frames=%lld\n",
    config.exposure_us, img.cols, img.rows, config.decimation, fps, cam_fps, (long long)frames);

  return 0;
}
