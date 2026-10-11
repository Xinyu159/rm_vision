// PnP 实时检查视图：相机 → YOLO → Solver 的满帧率链路 + 重投影叠加 + 浏览器可控制
//
// 干什么用的：
//   ① 看 PnP 准不准：目标放卷尺量好的距离上，读画面里的 d/yaw/pitch vs 真值；
//      青色重投影框（用解算位姿 + 定 pitch/roll 模型回投）应压在绿色检测框上，
//      err = 四点重投影残差（px）—— 偏大说明外参/内参/型别有一样不对
//   ② 看手眼标定有没有生效：外参是 yaml 里的占位值（理想正装）时叠加会印
//      "EXTR NOMINAL"，标定覆盖后变 "EXTR CALIB"
//
// 帧率：主循环 = 取帧 → 推理 → 解算，与实车部署同一条链、无 sleep 不降分辨率；
//       画面绘制 + MJPEG 推流在旁路拷贝上按 ~30fps 做（编码在服务线程），不拖管道。
//       画面上 FPS 是管道真实吞吐（读数不要只看这个数——它含绘制抖动）。
// IMU 未接入：R_gimbal2world = 单位阵 → world 系结果与 gimbal 系相同（相对相机）；
//       接入后每帧 solver->set_R_gimbal2world(q) 即世界系（标注后做）。
//
// 注意：画面叠加文字只能 ASCII——Hershey 字体渲染不了中文；参数必须用 "=" 传值
// 用法：
//   ./pnp_view --serve=8081               # 浏览器控制台（推荐；Save=存标注快照 / Quit）
//   ./pnp_view                            # 窗口模式（s 存标注快照 / q 退 / w 白平衡）
//   ./pnp_view --exposure=3000 --scale=0.75
// 快捷键（窗口模式）：q / ESC 退出；s 存标注快照；w 重做一次白平衡

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <deque>
#include <fmt/core.h>
#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "cli_common.hpp"
#include "io/galaxy/galaxy.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/control_server.hpp"
#include "tools/exiter.hpp"
#include "tools/fs_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const cv::String keys =
  "{help h usage ? |      | 打印帮助（参数必须用 = 传值）}"
  "{@config-path  | configs/standard3.yaml | 配置文件路径（galaxy: 相机 + yolo: 检测 + 外参）}"
  "{exposure e    | -1   | 曝光时间 µs（>0 才生效；查 PnP 用 3000 左右少糊）}"
  "{gain g        | -1   | 增益 dB（>=0 才生效）}"
  "{fps f         | -1   | 目标帧率 Hz（>0 才生效）}"
  "{roi           |      | ROI 尺寸 WxH，如 1024x768（不改则不填）}"
  "{decimation d  | 0    | 抽行抽列 1~2（>0 才生效）}"
  "{wb w          |      | 白平衡 Off/Once/Continuous（空 = 按 yaml）}"
  "{scale         | 1.0  | 显示缩放 0.1~1.0（画面小一圈、省 CPU）}"
  "{serve         | 0    | >0 时改在该端口开浏览器控制台（画面 + Save/Quit + 状态）}"
  "{snap-folder   | /tmp/pnp_view | 存标注快照目录（网页 Save / 窗口 s；编号续存）}";

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

double deg(double rad) { return rad * 180.0 / CV_PI; }

/// 画检测框（绿）+ 重投影框（青）+ 标签。坐标为原图坐标，按显示缩放换算。
/// 返回本帧已解算板的平均重投影残差（px）；没有解算成功的返回 -1
double draw_armors(cv::Mat & img, const std::list<auto_aim::Armor> & armors, double scale,
                   const auto_aim::Solver & solver)
{
  double err_sum = 0;
  int err_n = 0;
  for (const auto & a : armors) {
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (const auto & p : a.points)
      poly.emplace_back((int)(p.x * scale), (int)(p.y * scale));
    cv::polylines(img, poly, true, {0, 255, 0}, 2);
    for (const auto & p : poly) cv::circle(img, p, 5, {0, 0, 255}, -1);

    double err = -1;
    if (a.solved) {
      // 用解算位姿（xyz_in_world + yaw 精搜索结果）重投影回画面：
      // 定点叠在绿色检测框上 = 该位姿与图像自洽（yaw 搜索残差，含 pitch/roll 假设误差）
      auto reproj = solver.reproject_armor(a.xyz_in_world, a.ypr_in_world[0], a.type, a.name);
      std::vector<cv::Point> poly2;
      poly2.reserve(4);
      for (const auto & p : reproj)
        poly2.emplace_back((int)(p.x * scale), (int)(p.y * scale));
      cv::polylines(img, poly2, true, {255, 255, 0}, 2);
      err = 0;
      for (int i = 0; i < 4; i++) err += cv::norm(a.points[i] - reproj[i]);
      err /= 4;
      err_sum += err;
      err_n++;
    }

    std::string label = cv::format(
      "%s %s %s %.2f", auto_aim::COLORS[a.color].c_str(), auto_aim::ARMOR_NAMES[a.name].c_str(),
      auto_aim::ARMOR_TYPES[a.type].c_str(), a.confidence);
    if (a.solved)
      label += cv::format(
        "  d=%.2fm yaw%+.1f pit%+.1f err%.1fpx", a.ypd_in_world[2], deg(a.ypr_in_world[0]),
        deg(a.ypr_in_world[1]), err);
    cv::Point org = poly[0] + cv::Point(-40, -14);
    org.x = std::max(org.x, 2);
    org.y = std::max(org.y, 20);
    cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
  }
  return err_n > 0 ? err_sum / err_n : -1;
}

void draw_overlay(
  cv::Mat & img, double fps, double fps_avg, double cam_hz, double lat_ms, uint64_t n,
  std::size_t det, const Stat & st_read, const Stat & st_pre, const Stat & st_infer,
  const Stat & st_post, const Stat & st_solve, double err_px, bool extr_calibrated,
  const io::GalaxyConfig & cam, const std::string & snap_note)
{
  const cv::Rect box(0, 0, std::min(950, img.cols), std::min(150, img.rows));
  cv::Mat roi = img(box);
  cv::Mat black(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
  cv::addWeighted(roi, 0.35, black, 0.65, 0, roi);

  const auto green = cv::Scalar(0, 255, 0);
  const auto yellow = cv::Scalar(0, 255, 255);
  cv::putText(
    img, cv::format("FPS %.1f  det %zu", fps, det), {12, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.1, green,
    3);
  cv::putText(
    img, cv::format("(2s avg %.1f  cam %.1fHz)", fps_avg, cam_hz), {300, 38},
    cv::FONT_HERSHEY_SIMPLEX, 0.65, yellow, 2);
  cv::putText(
    img,
    cv::format(
      "read %.2f  pre %.2f  infer %.2f  post %.2f  solve %.3f ms (p50)", st_read.pct(0.5),
      st_pre.pct(0.5), st_infer.pct(0.5), st_post.pct(0.5), st_solve.pct(0.5)),
    {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.75, yellow, 2);

  std::string info = cv::format("lat %.1fms  frames %llu  err ", lat_ms, (unsigned long long)n);
  info += err_px >= 0 ? std::string(cv::format("%.1fpx", err_px)) : "--";
  cv::putText(img, info, {12, 113}, cv::FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2);
  // 外参状态：占位值（未标定）时醒目告警——标定覆盖后应显示 CALIB
  cv::putText(
    img, extr_calibrated ? "EXTR CALIB" : "EXTR NOMINAL (run handeye!)",
    {560, 113}, cv::FONT_HERSHEY_SIMPLEX, 0.7, extr_calibrated ? green : cv::Scalar(0, 0, 255), 2);

  std::string bottom =
    cam.exposure_us > 0 ? cv::format("exp %.0fus", cam.exposure_us) : "exp camera-default";
  if (cam.gain_db >= 0) bottom += cv::format(" gain %.1fdB", cam.gain_db);
  if (cam.decimation > 1) bottom += cv::format(" decim %d", cam.decimation);
  bottom += "   [q]uit [s]nap [w]WB";
  if (!snap_note.empty()) bottom += "   " + snap_note;
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
  if (!cli_tools::config_path_ok("pnp_view", config_path)) return 1;
  auto camera_config = io::galaxy_config_from_yaml(config_path);
  if (!cli_tools::apply_camera_overrides(cli, camera_config, "pnp_view")) return 1;

  double scale = cli.get<double>("scale");
  if (scale <= 0 || scale > 1) scale = 1.0;
  int port = cli.get<int>("serve");
  const bool serve_mode = port > 0;
  const std::string snap_folder = cli.get<cv::String>("snap-folder");
  if (!tools::mkdirs(snap_folder)) {
    tools::logger()->error("[pnp_view] 建快照目录失败: {}", snap_folder);
    return 1;
  }
  int snap_count = 0;
  while (::access(fmt::format("{}/snap_{}.jpg", snap_folder, snap_count + 1).c_str(), F_OK) == 0)
    snap_count++;
  std::string last_snap;

  // 解算链：检测 + PnP + yaw 精搜索（本工具的意义所在，默认全开）
  auto solver_config = auto_aim::solver_config_from_yaml(config_path);
  const bool extr_calibrated = solver_config.R_camera2gimbal != auto_aim::R_CAMERA2GIMBAL_NOMINAL;
  if (!extr_calibrated)
    tools::logger()->warn(
      "[pnp_view] R_camera2gimbal 还是理想正装占位值——画面会显示 EXTR NOMINAL；"
      "先跑手眼标定并覆盖 configs/standard3.yaml");

  std::unique_ptr<auto_aim::YOLO> yolo;
  std::unique_ptr<auto_aim::Solver> solver;
  try {
    yolo.reset(new auto_aim::YOLO(auto_aim::yolo_config_from_yaml(config_path)));
    solver.reset(new auto_aim::Solver(solver_config));
  } catch (const std::exception & e) {
    tools::logger()->error("[pnp_view] 载入检测/解算失败: {}", e.what());
    return 1;
  }

  tools::Exiter exiter;
  io::Galaxy camera(camera_config);

  cv::Mat raw, bgr, disp;
  std::chrono::steady_clock::time_point ts;
  for (int i = 0; i < 5; i++) camera.read(raw, ts);  // 预热
  if (raw.empty()) {
    tools::logger()->error("[pnp_view] 预热没拿到帧，退出");
    return 1;
  }
  yolo->detect(raw);  // 预热一次推理（首次含 CUDA 初始化，别让它污染统计）

  const std::string title = "rm_vision pnp view";
  tools::ControlServer cs;
  cs.show_auto = false;  // 本工具没有"自动存图"语义
  std::atomic<bool> quit{false};
  std::thread server;
  if (serve_mode) {
    server = std::thread(
      tools::serve_control, port, std::ref(cs), std::ref(quit), std::string("pnp_view"));
    tools::logger()->info(
      "[pnp_view] 浏览器控制台（端口 {}）：Save = 存标注快照，状态每 0.5s 刷新", port);
  } else {
    tools::logger()->info("[pnp_view] 窗口模式；无显示环境时请用 --serve=8081");
  }

  // 已解算板的平均重投影残差（只在绘制帧算，30Hz 足够看）
  double err_px = -1;
  auto do_snap = [&]() {
    if (disp.empty()) return;
    snap_count++;
    last_snap = fmt::format("{}/snap_{}.jpg", snap_folder, snap_count);
    cv::imwrite(last_snap, disp);
    tools::logger()->info("[pnp_view] 已存标注快照: {}", last_snap);
  };

  Stat st_read, st_pre, st_infer, st_post, st_solve, st_draw;
  std::deque<std::chrono::steady_clock::time_point> win;
  bool ever_visible = false;  // 窗口是否曾经可见过（区分启动竞态与真的被关掉）
  uint64_t n = 0, seen_web = 0;
  std::size_t det_count = 0;
  auto_aim::YOLO::Timings timings;
  auto t0 = std::chrono::steady_clock::now();
  auto t_last_log = t0, t_last_cam = t0, t_last_draw = t0, t_last_status = t0;
  auto t_last_snap = t0 - std::chrono::seconds(1);
  double fps = 0, fps_avg = 0, cam_hz = 0, lat_ms = 0;

  while (!exiter.exit() && !quit) {
    // ---- 管道：取帧 → 检测 → 解算（与实车部署同一条链，不打折）----
    auto t_before_read = std::chrono::steady_clock::now();
    camera.read(raw, ts);
    if (raw.empty()) continue;
    auto now = std::chrono::steady_clock::now();
    st_read.push(std::chrono::duration<double, std::milli>(now - t_before_read).count());

    // 延迟要在检测之前量（检测耗时 ~10ms，含进去就把画面延迟算多了）
    lat_ms = tools::delta_time(now, ts) * 1000.0;

    std::list<auto_aim::Armor> armors;
    try {
      armors = yolo->detect(raw);  // 传原始 Bayer，内部自己转换（省一次全幅 cvtColor）
      timings = yolo->last_timings();
    } catch (const std::exception & e) {
      tools::logger()->error("[pnp_view] 检测失败: {}", e.what());
      break;
    }
    det_count = armors.size();
    st_pre.push(timings.preprocess);
    st_infer.push(timings.inference);
    st_post.push(timings.postprocess);

    double solve_ms = 0;
    for (auto & a : armors) {
      solver->solve(a);  // 失败不改 a（solved 仍 false），标签就不带距离
      solve_ms += solver->last_timings().total();
    }
    st_solve.push(solve_ms);

    n++;
    win.push_back(now);
    while (!win.empty() && tools::delta_time(now, win.front()) > 2.0) win.pop_front();
    double span = win.empty() ? 0 : tools::delta_time(now, win.front());
    fps = span > 0.3 ? (win.size() - 1) / span : 0.0;
    fps_avg = tools::delta_time(now, t0) > 0.3 ? n / tools::delta_time(now, t0) : 0.0;
    if (tools::delta_time(now, t_last_cam) >= 1.0) {  // 相机自报值每秒查一次
      cam_hz = camera.actual_framerate();
      t_last_cam = now;
    }

    // ---- 旁路：绘制 + 推流按 ~30fps 做（编码在服务线程），不拖上面的管道 ----
    bool need_draw = tools::delta_time(now, t_last_draw) >= 0.033;
    if (need_draw) {
      auto t_draw0 = std::chrono::steady_clock::now();
      cv::cvtColor(raw, bgr, cv::COLOR_BayerBG2BGR);  // RGGB 布局 → BG 常量（见台账 §5）
      if (scale != 1.0)
        cv::resize(bgr, disp, {}, scale, scale, cv::INTER_AREA);
      else
        disp = bgr;
      err_px = draw_armors(disp, armors, scale, *solver);
      std::string snap_note = tools::delta_time(now, t_last_snap) < 1.2 ? "SNAPPED" : "";
      draw_overlay(
        disp, fps, fps_avg, cam_hz, lat_ms, n, det_count, st_read, st_pre, st_infer, st_post,
        st_solve, err_px, extr_calibrated, camera_config, snap_note);

      if (serve_mode) {
        cs.streamer.publish(disp);  // JPEG 编码在服务线程做（见 mjpeg_server.hpp 注释）
      }
      st_draw.push(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_draw0)
          .count());
      t_last_draw = now;
    }

    // 存快照请求：窗口/网页（cs.save_requests）统一消费
    uint64_t r2 = cs.save_requests.load();
    if (r2 > seen_web) {
      seen_web = r2;
      do_snap();
      t_last_snap = std::chrono::steady_clock::now();
    }

    if (!serve_mode && need_draw) {  // 同绘制节拍（disp 本拍必已画好，不会喂空图给 imshow）
      try {
        cv::imshow(title, disp);
        int k = cv::waitKey(1);
        if (k == 'q' || k == 27) {
          tools::logger()->info("[pnp_view] 按键退出");
          break;
        }
        if (k == 's') {
          do_snap();
          t_last_snap = std::chrono::steady_clock::now();
        }
        if (k == 'w') camera.white_balance_once();
        // 窗口可见性：GTK 下新窗口映射有延迟，首帧可能短暂不可见 → 只有"曾经可见过"才算被点 X 关掉
        if (cv::getWindowProperty(title, cv::WND_PROP_VISIBLE) >= 1) {
          ever_visible = true;
        } else if (ever_visible) {
          tools::logger()->info("[pnp_view] 窗口已关闭");
          break;
        }
      } catch (const cv::Exception & e) {
        tools::logger()->error("[pnp_view] 打不开窗口（{}）——无显示环境请用 --serve=8081", e.what());
        break;
      }
    }

    if (tools::delta_time(now, t_last_status) >= 0.5) {
      cs.set_status(fmt::format(
        "{{\"fps\":{:.1f},\"cam_hz\":{:.1f},\"frames\":{},\"det\":{},\"lat_ms\":{:.1f},"
        "\"read_ms\":{:.2f},\"infer_ms\":{:.2f},\"solve_ms\":{:.3f},\"err_px\":{:.2f},"
        "\"extr\":\"{}\",\"snap\":\"{}\"}}",
        fps, cam_hz, n, det_count, lat_ms, st_read.pct(0.5), st_infer.pct(0.5), st_solve.pct(0.5),
        err_px >= 0 ? err_px : -1.0, extr_calibrated ? "calib" : "nominal", last_snap));
      t_last_status = now;
    }

    if (tools::delta_time(now, t_last_log) >= 3.0) {
      tools::logger()->info(
        "[pnp_view] fps={:.1f} 相机自报={:.1f}Hz 延迟={:.1f}ms 帧数={} 检出={} err={:.2f}px | "
        "读帧 p50={:.2f}/p99={:.2f} 推理 p50={:.2f}/p99={:.2f} 解算 p50={:.3f}/p99={:.3f} "
        "绘制 p50={:.2f}/p99={:.2f} ms",
        fps, cam_hz, lat_ms, n, det_count, err_px, st_read.pct(0.5), st_read.pct(0.99),
        st_infer.pct(0.5), st_infer.pct(0.99), st_solve.pct(0.5), st_solve.pct(0.99),
        st_draw.pct(0.5), st_draw.pct(0.99));
      t_last_log = now;
    }
  }

  // 收尾统计（写进台账用）
  auto t_end = std::chrono::steady_clock::now();
  if (n > 0) {
    tools::logger()->info(
      "[pnp_view] 汇总：运行 {:.1f}s 帧数 {} 平均fps={:.1f} | 读帧 p50={:.2f}/p99={:.2f} "
      "预处理 p50={:.2f}/p99={:.2f} 推理 p50={:.2f}/p99={:.2f} 解码 p50={:.2f}/p99={:.2f} "
      "解算 p50={:.3f}/p99={:.3f} 绘制 p50={:.2f}/p99={:.2f} ms",
      tools::delta_time(t_end, t0), n, n / std::max(0.001, tools::delta_time(t_end, t0)),
      st_read.pct(0.5), st_read.pct(0.99), st_pre.pct(0.5), st_pre.pct(0.99), st_infer.pct(0.5),
      st_infer.pct(0.99), st_post.pct(0.5), st_post.pct(0.99), st_solve.pct(0.5),
      st_solve.pct(0.99), st_draw.pct(0.5), st_draw.pct(0.99));
  }

  quit = true;
  if (cs.quit_requested) tools::logger()->info("[pnp_view] 网页按了 Quit");
  if (server.joinable()) server.join();
  tools::logger()->info("[pnp_view] 退出");
  return 0;
}
