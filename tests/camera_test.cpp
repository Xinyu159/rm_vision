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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "cli_common.hpp"
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
  "{wb w          |      | 白平衡 Off/Once/Continuous（空 = 按 yaml）}"
  "{seconds s     | 3    | 测量时长（秒）}"
  "{save          |      | 预热后存一帧到该路径（Bayer→BGR 的 png）}"
  "{save-raw      |      | 存原始 Bayer 帧（8bit 灰度 png，判阵列用）}"
  "{save-all      |      | 存 4 种 Bayer 解码变体 <前缀>_RG/BG/GR/GB.png（判阵列用）}";

int main(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<cv::String>("@config-path");
  if (!cli_tools::config_path_ok("camera_test", config_path)) return 1;
  auto config = io::galaxy_config_from_yaml(config_path);
  if (!cli_tools::apply_camera_overrides(cli, config, "camera_test")) return 1;
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
    if (img.channels() == 1) {
      // 传感器是 RGGB（左上角=红，相机自报 "BayerRG8"），但 OpenCV 的 Bayer 命名
      // 看的是第二行第二列：RGGB 布局要用 COLOR_BayerBG2BGR！用 RG 会红蓝互换。
      // 2026-10-08 纯红屏实测定案：BG 变体 → 红；RG 变体 → 蓝。
      cv::cvtColor(img, bgr, cv::COLOR_BayerBG2BGR);
    } else {
      bgr = img;
    }
    cv::imwrite(save, bgr);
    tools::logger()->info("[camera_test] 已存帧: {}", (std::string)save);
  }

  // Bayer 阵列判型：原始帧 + 4 种解码变体 + 中心区子阵均值
  cv::String save_raw = cli.get<cv::String>("save-raw");
  if (!save_raw.empty() && !img.empty()) {
    cv::imwrite(save_raw, img);
    tools::logger()->info("[camera_test] 已存原始 Bayer: {}", (std::string)save_raw);
  }

  cv::String save_all = cli.get<cv::String>("save-all");
  if (!save_all.empty() && !img.empty() && img.channels() == 1) {
    const struct
    {
      const char * name;
      int code;
    } patterns[] = {
      {"RG", cv::COLOR_BayerRG2BGR},
      {"BG", cv::COLOR_BayerBG2BGR},
      {"GR", cv::COLOR_BayerGR2BGR},
      {"GB", cv::COLOR_BayerGB2BGR}};
    for (const auto & p : patterns) {
      cv::Mat bgr;
      cv::cvtColor(img, bgr, p.code);
      std::string path = (std::string)save_all + "_" + p.name + ".png";
      cv::imwrite(path, bgr);
      tools::logger()->info("[camera_test] 已存变体: {}", path);
    }
  }

  if (!img.empty() && img.channels() == 1) {
    // 中心 256x256 的 4 个 Bayer 子阵均值（避开边缘渐晕；[00] 是画面左上角像素所在子阵）
    int cw = std::min(256, img.cols), chh = std::min(256, img.rows);
    cv::Rect r((img.cols - cw) / 2, (img.rows - chh) / 2, cw, chh);
    double sum[2][2] = {{0, 0}, {0, 0}};
    long cnt[2][2] = {{0, 0}, {0, 0}};
    for (int y = r.y; y < r.y + r.height; y++) {
      const uint8_t * rowp = img.ptr<uint8_t>(y);
      for (int x = r.x; x < r.x + r.width; x++) {
        sum[y % 2][x % 2] += rowp[x];
        cnt[y % 2][x % 2]++;
      }
    }
    tools::logger()->info(
      "[camera_test] 中心 {}x{} 子阵均值: [00]={:.1f} [01]={:.1f} [10]={:.1f} [11]={:.1f}", cw, chh,
      sum[0][0] / cnt[0][0], sum[0][1] / cnt[0][1], sum[1][0] / cnt[1][0], sum[1][1] / cnt[1][1]);
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
