// 标定采集：相机图 + 云台四元数成对存盘（原图 {n}.jpg + 四元数 {n}.txt，1 起编号、wxyz）
//
// 三种控制方式可同时开：
//   终端（默认）：回车 / s = 存一张，q = 退出（stdin 独立线程，无终端时自动停用不炸）
//   浏览器（--serve=8080）：http://<Jetson IP>:8080/ 控制台 —— 实时画面 + Save/Auto/Quit/状态，
//                           VS Code 关了也能全流程操作（页面每 1s 轮询 /status）
//   窗口（--display=1）：本机带显示器时开窗，s 存 q 退
//
// 画面叠加（全英文 ASCII——Hershey 字体不支持中文，中文会渲染成 "??"）：
//   帧率 / 延迟 / 张数 + 云台欧拉角 + 棋盘识别结果（board OK / -- / ...）
// 识别在后台线程跑（半分辨率 findChessboardCornersSB）：主循环只管采图/存图/推流，
// 检测慢也不掉帧；画面提示比实际晚 ~0.2s，存图用的是原图，不受影响
//
// 注意：参数必须用 "=" 传值（--serve=8080），空格分隔会把数值当成配置文件路径
// 注意：下位机链路断时 gimbal->q() 会阻塞（同济同语义）——疑似断线先用 --no-q 采图；
//       万一卡在 q() 里（Exiter 的 Ctrl+C 只置标志、救不了阻塞点）需另开终端 kill
// 用法：
//   ./capture --serve=8080                     # 浏览器控制台（推荐，边看边存）
//   ./capture                                  # 纯终端（回车存图）
//   ./capture --display=1                      # 开窗口看
//   ./capture --no-q --serve=8080              # 下位机不在：只采图（不写四元数）
//   ./capture --auto=3 --serve=8080            # 每 3s 自动存一张（网页可随时改/关）
//   ./capture --output-folder=assets/img_with_q

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <deque>
#include <Eigen/Geometry>
#include <fmt/core.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/control_server.hpp"
#include "tools/exiter.hpp"
#include "tools/fs_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace
{

const cv::String keys =
  "{help h usage ? |      | 打印帮助（本程序参数必须用 = 传值）}"
  "{@config-path  | configs/calibration.yaml | 配置文件路径（相机 + 串口 + 棋盘格）}"
  "{output-folder o | assets/img_with_q | 输出目录（编号.jpg 原图 + 编号.txt 四元数 wxyz）}"
  "{serve         | 0    | >0：该端口开浏览器控制台（画面 + Save/Auto/Quit + 状态）}"
  "{display       | 0    | 1：额外开本地窗口（s 存 / q 退；需显示器）}"
  "{no-q          | 0    | 1：不连下位机、只采图（不写四元数）}"
  "{auto          | 0    | >0：每 N 秒自动存一张（网页 Auto 可随时改/关）}"
  "{scale         | 1.0  | 显示缩放 0.1~1.0（画面小一圈、省 CPU）}";

/// OpenCV 解析器把裸 `--flag` 存成字符串 "true"（get<int> 解析 "true" 失败会静默变 0！）——
/// 统一按字符串判：--no-q / --no-q=1 / --no-q=true 都生效，--no-q=0 关闭
bool flag_on(const cv::CommandLineParser & cli, const std::string & name)
{
  cv::String v = cli.get<cv::String>(name);
  return v == "true" || v == "1" || v == "yes";
}

/// 棋盘识别（SB 变体 + CLAHE 对比度增强）。经典 findChessboardCorners 在无板时实测
/// 4.3s/张（自适应阈值），曾把主循环拖到 0.2fps；SB 半分辨率只要 ~130ms，故全程改用 SB。
/// CLAHE 依据（2026-10-11 暗场地实拍）：半分辨率 plain SB 两张测试图全挂、加 CLAHE 后全过；
/// 离线两程序同样带 CLAHE 兜底链（全分辨率优先、暗图退化半分辨率）。
bool detect_board_sb(
  const cv::Mat & bgr, const cv::Size & pattern_size, std::vector<cv::Point2f> & corners)
{
  cv::Mat gray, enhanced;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::createCLAHE(3.0, cv::Size(8, 8))->apply(gray, enhanced);
  corners.clear();
  return cv::findChessboardCornersSB(enhanced, pattern_size, corners);
}

/// 棋盘检测工作线程：主循环只"放入半分辨率副本 + 取最近结果"，检测绝不阻塞采集/存图/推流。
/// 结果角点 ×2 还原到全分辨率坐标（显示与状态用）。
struct BoardDetector
{
  cv::Size pattern_size;
  double scale = 0.5;  // 检测用缩放：角点坐标 /scale 还原到全分辨率

  void start() { thread = std::thread([this] { run(); }); }
  void stop()
  {
    quit = true;
    cond.notify_all();
    if (thread.joinable()) thread.join();
  }

  /// 主循环：放入一帧（上一帧还没检测完就跳过，不排队、不堆积）
  void submit(const cv::Mat & bgr)
  {
    std::lock_guard<std::mutex> lock(m);
    if (in_seq != done_seq) return;
    cv::resize(bgr, in, {}, scale, scale, cv::INTER_AREA);
    in_seq++;
    cond.notify_all();
  }

  /// 主循环：最近一次结果（corners 已还原到全分辨率）；还没有任何结果返回 false
  bool latest(std::vector<cv::Point2f> & corners_full, bool & ok_out, double & age_s)
  {
    std::lock_guard<std::mutex> lock(m);
    if (!has_result) return false;
    corners_full.resize(corners.size());
    for (std::size_t i = 0; i < corners.size(); i++) corners_full[i] = corners[i] / (float)scale;
    ok_out = ok;
    age_s = tools::delta_time(std::chrono::steady_clock::now(), result_time);
    return true;
  }

private:
  std::mutex m;
  std::condition_variable cond;
  cv::Mat in;  // 待检测帧（半分辨率 BGR）
  uint64_t in_seq = 0, done_seq = 0;
  bool ok = false, has_result = false;
  std::vector<cv::Point2f> corners;  // 半分辨率坐标
  std::chrono::steady_clock::time_point result_time{};
  std::atomic<bool> quit{false};
  std::thread thread;

  void run()
  {
    while (!quit) {
      cv::Mat frame;
      {
        std::unique_lock<std::mutex> lock(m);
        cond.wait_for(
          lock, std::chrono::milliseconds(100), [&] { return in_seq != done_seq || quit.load(); });
        if (in_seq == done_seq) continue;
        frame = in.clone();  // 深拷贝：submit 的 resize 会原地复用 in 的缓冲
        done_seq = in_seq;
      }
      std::vector<cv::Point2f> cs;
      bool ok_now = detect_board_sb(frame, pattern_size, cs);
      {
        std::lock_guard<std::mutex> lock(m);
        ok = ok_now;
        corners = cs;
        has_result = true;
        result_time = std::chrono::steady_clock::now();
      }
    }
  }
};

struct OverlayInfo
{
  double fps = 0;
  double lat_ms = 0;
  uint64_t frames = 0;
  int count = 0;  // 已存总张数
  bool has_q = false;
  Eigen::Vector3d ypr = Eigen::Vector3d::Zero();  // 云台欧拉角（deg）
  std::string board_text = "board ...";           // board OK / board -- / board ...（检测中）
  int auto_sec = 0;
  bool saved_flash = false;  // 刚存图：绿色闪一行提示
  int saved_no = 0;
};

void draw_overlay(cv::Mat & img, const OverlayInfo & info)
{
  const cv::Rect box(0, 0, std::min(700, img.cols), std::min(136, img.rows));
  cv::Mat roi = img(box);
  cv::Mat black(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
  cv::addWeighted(roi, 0.35, black, 0.65, 0, roi);

  const auto green = cv::Scalar(0, 255, 0);
  const auto yellow = cv::Scalar(0, 255, 255);
  cv::putText(
    img,
    cv::format(
      "FPS %.1f  lat %.1fms  frames %llu  saved %d", info.fps, info.lat_ms,
      (unsigned long long)info.frames, info.count),
    {12, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.8, green, 2);

  if (info.has_q)
    cv::putText(
      img, cv::format("gimbal yaw %.1f  pitch %.1f  roll %.1f (deg)", info.ypr[0], info.ypr[1], info.ypr[2]),
      {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2);
  else
    cv::putText(
      img, "NO-Q mode: image only (no quaternion)", {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.8, yellow,
      2);

  std::string line3 = info.board_text;
  if (info.auto_sec > 0) line3 += cv::format("   AUTO %ds", info.auto_sec);
  if (info.saved_flash) line3 += cv::format("   SAVED %d", info.saved_no);
  cv::putText(
    img, line3, {12, 120}, cv::FONT_HERSHEY_SIMPLEX, 0.8, info.saved_flash ? green : yellow, 2);
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
  if (config_path.find('/') == std::string::npos && config_path.find(".yaml") == std::string::npos) {
    tools::logger()->error(
      "[capture] 位置参数 \"{}\" 不像配置文件路径（参数必须用 = 传值，如 --serve=8080）",
      config_path);
    return 1;
  }
  const std::string output_folder = cli.get<cv::String>("output-folder");
  const bool has_q = !flag_on(cli, "no-q");
  const bool display = flag_on(cli, "display");
  int port = cli.get<int>("serve");
  if (port < 0 || port > 65535) port = 0;
  const bool serve_mode = port > 0;
  double scale = cli.get<double>("scale");
  if (scale <= 0 || scale > 1) scale = 1.0;

  // 棋盘内角点数（画面叠加识别用；语义见 configs/calibration.yaml 注释）
  auto yaml = tools::load(config_path);
  cv::Size pattern_size(
    tools::read<int>(yaml, "pattern_cols"), tools::read<int>(yaml, "pattern_rows"));
  tools::logger()->info(
    "[capture] 棋盘内角点 {}x{}（pattern_cols/pattern_rows；全 failure 先怀疑这里）",
    pattern_size.width, pattern_size.height);

  // 输出目录：递归建；已有图则从最大编号续采（不覆盖已有文件）
  if (!tools::mkdirs(output_folder)) {
    tools::logger()->error("[capture] 建输出目录失败: {}", output_folder);
    return 1;
  }
  int count = 0;
  while (::access(fmt::format("{}/{}.jpg", output_folder, count + 1).c_str(), F_OK) == 0) count++;
  if (count > 0)
    tools::logger()->warn(
      "[capture] 输出目录已有 {} 张，从 {}.jpg 继续（不覆盖已有文件）", count, count + 1);

  io::Camera camera(config_path);
  cv::Mat raw, img, disp;
  std::chrono::steady_clock::time_point ts;
  for (int i = 0; i < 5; i++) camera.read(raw, ts);  // 预热
  if (raw.empty()) {
    tools::logger()->error("[capture] 相机没出图（确认没有别的程序占着相机）");
    return 1;
  }

  std::unique_ptr<io::Gimbal> gimbal;
  if (has_q) {
    try {
      gimbal = std::make_unique<io::Gimbal>(config_path);
    } catch (const std::exception & e) {
      tools::logger()->error(
        "[capture] 下位机未就绪（{}）——插好电控板，或加 --no-q 只采图（不写四元数）", e.what());
      return 1;
    }
    tools::logger()->info("[capture] 下位机已连（四元数按相机时间戳插值；txt 顺序 wxyz）");
  } else {
    tools::logger()->warn("[capture] --no-q：只采图不写四元数（这种图不能用于手眼标定）");
  }

  // 棋盘检测工作线程（放在所有"会 return 1"的初始化之后：线程起了就不许中途 return）
  BoardDetector det;
  det.pattern_size = pattern_size;
  det.start();

  tools::Exiter exiter;
  tools::ControlServer cs;
  std::atomic<bool> quit{false};
  std::atomic<uint64_t> save_req{0};

  int auto_sec = cli.get<int>("auto");
  if (auto_sec < 0) auto_sec = 0;
  cs.auto_sec = auto_sec;
  if (auto_sec > 0) tools::logger()->info("[capture] 自动存图：每 {}s 一张（网页可改）", auto_sec);

  std::thread server;
  if (serve_mode) {
    server = std::thread(
      tools::serve_control, port, std::ref(cs), std::ref(quit), std::string("capture"));
  }

  // 终端控制（stdin 独立线程；isatty 才开，后台/管道场景自动跳过）
  if (::isatty(0)) {
    std::thread([&] {
      char ch = 0;
      while (!quit) {
        ssize_t nr = ::read(0, &ch, 1);
        if (nr <= 0) {
          tools::logger()->info("[capture] stdin 已关闭，终端控制停用（用网页或窗口）");
          break;
        }
        if (ch == '\n' || ch == 's')
          save_req++;
        else if (ch == 'q')
          quit = true;
      }
    }).detach();  // 故意 detach：退出时多半还阻塞在 read(0)，随进程一起消失
  } else {
    tools::logger()->info("[capture] stdin 非终端，终端控制停用（--serve 网页 / --display 窗口）");
  }

  const std::string win_title = "rm_vision capture [s]ave [q]uit";
  bool window_visible = false;
  std::deque<std::chrono::steady_clock::time_point> win;
  uint64_t n = 0, seen_req = 0, seen_web = 0;
  int saved = 0, last_saved_no = 0;
  double fps = 0, lat_ms = 0;
  auto now = std::chrono::steady_clock::now();
  auto t_last_submit = now - std::chrono::seconds(1);  // 首帧立即送检
  auto t_last_pub = now, t_last_auto = now, t_last_status = now;
  auto t_last_log = now, t_last_save = now;

  while (!exiter.exit() && !quit) {
    camera.read(raw, ts);
    if (raw.empty()) continue;
    cv::cvtColor(raw, img, cv::COLOR_BayerBG2BGR);  // RGGB 布局 → BG 常量（台账 §5）
    now = std::chrono::steady_clock::now();

    Eigen::Quaterniond q_cur = Eigen::Quaterniond::Identity();
    if (gimbal) q_cur = gimbal->q(ts);  // 相机时间戳处插值（链路断会阻塞，见文件头注释）

    // 帧率 / 延迟
    n++;
    lat_ms = tools::delta_time(now, ts) * 1000.0;
    win.push_back(now);
    while (!win.empty() && tools::delta_time(now, win.front()) > 2.0) win.pop_front();
    double span = win.size() > 1 ? tools::delta_time(now, win.front()) : 0;
    fps = span > 0.3 ? (win.size() - 1) / span : 0.0;

    // 棋盘检测：~5Hz 把半分辨率副本交给工作线程（上一张没检测完就跳过，不排队）
    if (tools::delta_time(now, t_last_submit) >= 0.2) {
      det.submit(img);
      t_last_submit = now;
    }
    std::vector<cv::Point2f> corners;  // 已还原到全分辨率
    bool board_ok_raw = false;
    double board_age = 1e9;
    bool have_board = det.latest(corners, board_ok_raw, board_age);
    bool board_fresh = have_board && board_age < 1.5;  // >1.5s 无新结果 = 检测卡住
    bool board_ok = board_fresh && board_ok_raw;

    // 显示副本：窗口模式每帧画；推流模式 ~30fps 画（JPEG 编码在服务线程做，见 mjpeg_server.hpp）
    bool pub_due = serve_mode && tools::delta_time(now, t_last_pub) >= 0.033;
    if (display || pub_due) {
      if (scale != 1.0)
        cv::resize(img, disp, {}, scale, scale, cv::INTER_AREA);
      else
        disp = img.clone();  // 必须 clone：disp 会画角点圈+叠加文字，别名会把这层污染
                             // 直接带进随后 imwrite 的 img 原图（2026-10-11 实测：圈线
                             // 让离线检测挂掉、live 却正常，浪费了 1/4 采集帧）
      if (board_fresh) {  // 过期的角点不画（会落在错位置）
        if (scale != 1.0) {
          std::vector<cv::Point2f> corners_scaled;
          corners_scaled.reserve(corners.size());
          for (const auto & pt : corners) corners_scaled.push_back(pt * (float)scale);
          cv::drawChessboardCorners(disp, pattern_size, corners_scaled, board_ok_raw);
        } else {
          cv::drawChessboardCorners(disp, pattern_size, corners, board_ok_raw);
        }
      }

      OverlayInfo info;
      info.fps = fps;
      info.lat_ms = lat_ms;
      info.frames = n;
      info.count = count;
      info.has_q = has_q;
      if (has_q) info.ypr = tools::eulers(q_cur, 2, 1, 0) * 57.3;
      info.board_text =
        !board_fresh ? "board ..." : (board_ok_raw ? "board OK" : "board --");
      info.auto_sec = cs.auto_sec.load();
      info.saved_flash = tools::delta_time(now, t_last_save) < 0.6;
      info.saved_no = last_saved_no;
      draw_overlay(disp, info);
    }
    if (pub_due) {
      cs.streamer.publish(disp);
      t_last_pub = now;
    }

    if (display) {
      try {
        cv::imshow(win_title, disp);
        int k = cv::waitKey(1);
        if (k == 'q' || k == 27)
          quit = true;
        else if (k == 's')
          save_req++;
        if (cv::getWindowProperty(win_title, cv::WND_PROP_VISIBLE) >= 1) {
          window_visible = true;
        } else if (window_visible) {
          tools::logger()->info("[capture] 窗口已关闭");
          quit = true;
        }
      } catch (const cv::Exception & e) {
        tools::logger()->error("[capture] 开窗口失败（{}）——改用 --serve=8080 浏览器版", e.what());
        quit = true;
      }
    }

    // 存图请求汇总：终端/窗口（save_req）+ 网页（cs.save_requests）+ 自动（cs.auto_sec）
    uint64_t r1 = save_req.load(), r2 = cs.save_requests.load();
    bool want_save = r1 > seen_req || r2 > seen_web;
    seen_req = r1;
    seen_web = r2;
    int cur_auto = cs.auto_sec.load();
    if (cur_auto < 0) cur_auto = 0;
    if (cur_auto > 0 && tools::delta_time(now, t_last_auto) >= cur_auto) {
      want_save = true;
      t_last_auto = now;
    }
    if (want_save) {
      count++;
      saved++;
      last_saved_no = count;
      t_last_save = now;
      auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
      cv::imwrite(img_path, img);
      if (gimbal) {
        auto q_path = fmt::format("{}/{}.txt", output_folder, count);
        std::ofstream q_file(q_path);
        q_file << fmt::format("{} {} {} {}", q_cur.w(), q_cur.x(), q_cur.y(), q_cur.z()) << "\n";
        tools::logger()->info(
          "[capture] 已存 {}.jpg + {}.txt（q wxyz {:.4f} {:.4f} {:.4f} {:.4f}）", count, count,
          q_cur.w(), q_cur.x(), q_cur.y(), q_cur.z());
      } else {
        tools::logger()->info("[capture] 已存 {}.jpg（no-q：无四元数）", count);
      }
    }

    // 网页状态（0.5s 一次；纯 ASCII 值，JSON）
    if (tools::delta_time(now, t_last_status) >= 0.5) {
      cs.set_status(fmt::format(
        "{{\"frames\":{},\"saved\":{},\"session\":{},\"fps\":{:.1f},\"lat_ms\":{:.1f},"
        "\"has_q\":{},\"auto_sec\":{},\"board\":{},\"out\":\"{}\"}}",
        n, count, saved, fps, lat_ms, has_q, cs.auto_sec.load(), board_ok, output_folder));
      t_last_status = now;
    }

    if (tools::delta_time(now, t_last_log) >= 3.0) {
      tools::logger()->info(
        "[capture] fps={:.1f} lat={:.1f}ms frames={} saved={} board={}", fps, lat_ms, n, count,
        board_ok ? "OK" : "--");
      t_last_log = now;
    }
  }

  quit = true;
  det.stop();
  if (cs.quit_requested) tools::logger()->info("[capture] 网页按了 Quit");
  if (server.joinable()) server.join();
  tools::logger()->info(
    "[capture] 退出：本次存 {} 张（累计 {}），目录 {}", saved, count, output_folder);
  if (saved == 0)
    tools::logger()->warn("[capture] 本次没存图——终端回车（或 s）/ 网页点 Save / --auto=N");
  if (has_q)
    tools::logger()->warn("[capture] 四元数在 {{n}}.txt（顺序 wxyz），可直接喂手眼标定程序");
  return 0;
}
