// 云台跟随闭环（最小版）：相机 → YOLO → Solver → 指令校验 → 串口发云台（mode=1 控制不开火）
//
// 干什么用的：把 PnP 解算出的目标角度闭环发给电控，云台应指向目标（目标移到哪跟到哪）。
// 与 pnp_view 的关系：同一条检测/解算链，但 pnp_view 只看不算（+IMU 不接）；本程序用
// 电控回报的 IMU 四元数把解算转到世界系，并按同济语义下发角度。
//
// 发送语义（对齐同济 sp_vision_25：aimer 输出 → gimbal.send；协议见 docs/serial-protocol.md）：
//   世界系绝对角：yaw = atan2(y_world, x_world)（x 前 y 左，左正）；pitch 向上为负。
//   角度基准每帧与电控同源：solver->set_R_gimbal2world(gimbal.q(相机时间戳))。
//   本版无 tracker、无弹道：yaw/pitch 就是目标中心视线方向（不做板面法向修正/重力补偿）。
//
// 指令校验链（每帧；任何一项不合格就不发新值，NAN 永不出口）：
//   ① 板须解算成功（solved），坐标须有限   ② pitch 限幅、yaw 归一（limit_rad）+ 上限
//   ③ 跳变闸：与上一有效指令差超阈（默认 yaw 60°/pitch 28°）先保持旧指令，连续 3 帧才
//      接受新值——单帧离群解（斜视翻解/误检）不会让云台乱甩   ④ 丢目标 >0.5s 发 mode=0
//      放弃控制（之后收尾行为由电控定）；重捕获从头过校验 ⑤ --send 默认 0（dry-run 只
//      打印不下发）：先用 dry 看 cmd vs gmb（闭环误差）数值合理，再 --send=1 真发。
//   发送帧 CRC 由 io::Gimbal 填（协议字节级自检：serial_test --mode=loop）。
//
// 注意：Gimbal 构造会等电控首帧（standard3.yaml first_q_timeout_s=0 无限等——现场先开电控）；
//      画面叠加文字只能 ASCII；参数必须用 "=" 传值
// 提速两件套（2026-10-11 场测）：--draw=0 关每帧绘制/推流（Save 时按需绘一帧落盘）；
//   动态 ROI 默认开（--det-roi=1）：按上一帧目标框扩边（默认 2.5 倍、最小边 256px）
//   只送 ROI 子区给检测——预处理只做一小块、网络输入里目标占比更大；ROI 内没检出
//   同帧回退全图，连丢 3 帧回全图重找（坐标回填由 yolo.detect(img,roi) 做）。
//
// 用法：
//   ./gimbal_follow --serve=8082              # dry-run：不发指令，看数值
//   ./gimbal_follow --serve=8082 --send=1     # 真发：云台跟随（先确认周围安全）
//   ./gimbal_follow --serve=8082 --send=1 --draw=0   # 场上测跟随：关绘制/推流追帧率
//   ./gimbal_follow --send=1 --pitch-limit-deg=30 --scale=0.75
// 快捷键（窗口模式）：q / ESC 退出；s 存标注快照；w 重做一次白平衡

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fmt/core.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "cli_common.hpp"
#include "io/galaxy/galaxy.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/control_server.hpp"
#include "tools/exiter.hpp"
#include "tools/fs_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const cv::String keys =
  "{help h usage ? |      | 打印帮助（参数必须用 = 传值，如 --send=1）}"
  "{@config-path  | configs/standard3.yaml | 配置（galaxy 相机 + yolo 检测 + 外参 + com_port）}"
  "{exposure e    | -1   | 曝光时间 µs（>0 才生效）}"
  "{gain g        | -1   | 增益 dB（>=0 才生效）}"
  "{fps f         | -1   | 目标帧率 Hz（>0 才生效）}"
  "{roi           |      | ROI 尺寸 WxH，如 1024x768（不改则不填）}"
  "{decimation d  | 0    | 抽行抽列 1~2（>0 才生效）}"
  "{wb w          |      | 白平衡 Off/Once/Continuous（空 = 按 yaml）}"
  "{scale         | 1.0  | 显示缩放 0.1~1.0（仅 draw=1 时有意义）}"
  "{draw          | 1    | 1 = 每帧画叠加 + 推流（~30fps）；0 = 关（追帧率；Save 快照按需绘一帧）}"
  "{det-roi       | 1    | 1 = 动态检测 ROI（按上一帧目标框扩边急裁）；0 = 全图检测}"
  "{det-roi-scale | 2.5  | det-roi=1 时 ROI 相对目标框的扩边倍数（>=1；另有 256px 最小边保底）}"
  "{serve         | 0    | >0 时在该端口开浏览器控制台（Quit 退出 / Save 存快照）}"
  "{send          | 0    | 0 = dry-run 只打印不下发；1 = 真发（先 dry 确认数值再开）}"
  "{pitch-limit-deg | 45 | pitch 指令限幅（±deg），超限本帧保持旧指令}"
  "{yaw-limit-deg | 180  | |yaw| 指令上限（deg）；云台机械限位小于 360° 时调小}"
  "{yaw-jump-deg  | 60   | 跳变闸：yaw 指令跳变阈值（deg）}"
  "{pitch-jump-deg | 28  | 跳变闸：pitch 指令跳变阈值（deg）}"
  "{snap-folder   | /tmp/gimbal_follow | Save 快照目录（编号续存）}";

namespace
{

double deg(double rad) { return rad * 180.0 / CV_PI; }

/// 叠加显示收集（全 ASCII——Hershey 渲染不了中文）
struct Display
{
  bool send_enabled = false;
  double fps = 0;
  std::size_t det = 0;
  double cmd_yaw = 0, cmd_pitch = 0, cmd_d = -1;  // 本帧指令（度；d 为目标距离 m，-1=无目标）
  double gmb_yaw = 0, gmb_pitch = 0;              // 电控回报的云台角（度）
  std::string state = "IDLE";
  uint64_t n_sent = 0, n_hold = 0, n_nan = 0, n_range = 0, n_jump = 0, n_release = 0;
  std::string snap_note;
};

void draw_overlay(cv::Mat & img, const Display & d)
{
  const cv::Rect box(0, 0, std::min(1100, img.cols), std::min(150, img.rows));
  cv::Mat roi = img(box);
  cv::Mat black(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
  cv::addWeighted(roi, 0.35, black, 0.65, 0, roi);

  const auto green = cv::Scalar(0, 255, 0);
  const auto yellow = cv::Scalar(0, 255, 255);
  const auto red = cv::Scalar(0, 0, 255);

  cv::putText(
    img, cv::format("FOLLOW %s   fps %.1f  det %zu", d.send_enabled ? "SEND=ON" : "DRY(no send)",
                    d.fps, d.det),
    {12, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, d.send_enabled ? red : green, 3);

  if (d.cmd_d > 0) {
    double cerr_y = d.cmd_yaw - d.gmb_yaw, cerr_p = d.cmd_pitch - d.gmb_pitch;
    while (cerr_y > 180) cerr_y -= 360;
    while (cerr_y <= -180) cerr_y += 360;
    cv::putText(
      img,
      cv::format(
        "cmd %+.1f/%+.1f  gmb %+.1f/%+.1f  cerr %+.1f/%+.1f (deg)  d %.2fm", d.cmd_yaw, d.cmd_pitch,
        d.gmb_yaw, d.gmb_pitch, cerr_y, cerr_p, d.cmd_d),
      {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2);
  } else {
    cv::putText(
      img, "cmd --/--   (no target)", {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2);
  }

  std::string line3 = cv::format(
    "state %s | sent %lu hold %lu drop nan %lu rng %lu jmp %lu rel %lu", d.state.c_str(),
    (unsigned long)d.n_sent, (unsigned long)d.n_hold, (unsigned long)d.n_nan,
    (unsigned long)d.n_range, (unsigned long)d.n_jump, (unsigned long)d.n_release);
  cv::putText(img, line3, {12, 118}, cv::FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2);

  std::string bottom = "   [q]uit [s]nap";
  if (!d.snap_note.empty()) bottom += "   " + d.snap_note;
  cv::putText(img, bottom, {12, 146}, cv::FONT_HERSHEY_SIMPLEX, 0.65, yellow, 2);
}

/// 画检测框（绿）+ 目标框（红粗）+ 距离标签
void draw_armors(
  cv::Mat & img, const std::list<auto_aim::Armor> & armors, const auto_aim::Armor * tgt,
  double scale)
{
  for (const auto & a : armors) {
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (const auto & p : a.points)
      poly.emplace_back((int)(p.x * scale), (int)(p.y * scale));
    bool is_tgt = (&a == tgt);
    cv::polylines(img, poly, true, is_tgt ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0),
                  is_tgt ? 3 : 2);
    if (a.solved) {
      cv::putText(
        img,
        cv::format(
          "%s %.2fm", auto_aim::ARMOR_NAMES[a.name].c_str(),
          std::sqrt(
            a.xyz_in_world[0] * a.xyz_in_world[0] + a.xyz_in_world[1] * a.xyz_in_world[1] +
            a.xyz_in_world[2] * a.xyz_in_world[2])),
        poly[0] + cv::Point(-40, -14), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        is_tgt ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255), 2);
    }
  }
}

/// 由目标框（或上一 ROI）扩边出下帧检测窗：绕中心放大 scale 倍、最小边 256px，
/// 裁进图像并量化（x/y 偶对齐 Bayer 相位；w/h 32 对齐让 letterbox 尺寸稳定少重清黑边）
cv::Rect expand_det_roi(const cv::Mat & img, const cv::Rect & box, double scale)
{
  const double cx = box.x + box.width / 2.0, cy = box.y + box.height / 2.0;
  int w = std::min(std::max((int)std::lround(box.width * scale), 256), img.cols);
  int h = std::min(std::max((int)std::lround(box.height * scale), 256), img.rows);
  int x = std::min(std::max((int)std::lround(cx - w / 2.0), 0), img.cols - w);
  int y = std::min(std::max((int)std::lround(cy - h / 2.0), 0), img.rows - h);
  x &= ~1;
  y &= ~1;
  w = std::min((w + 31) & ~31, img.cols - x);
  h = std::min((h + 31) & ~31, img.rows - y);
  return {x, y, w, h};
}

}  // namespace

int main(int argc, char ** argv)
{
  std::signal(SIGPIPE, SIG_IGN);

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<cv::String>("@config-path");
  if (!cli_tools::config_path_ok("gimbal_follow", config_path)) return 1;
  auto camera_config = io::galaxy_config_from_yaml(config_path);
  if (!cli_tools::apply_camera_overrides(cli, camera_config, "gimbal_follow")) return 1;

  double scale = cli.get<double>("scale");
  if (scale <= 0 || scale > 1) scale = 1.0;
  const bool draw_enabled = cli.get<int>("draw") != 0;
  const bool det_roi_enabled = cli.get<int>("det-roi") != 0;
  double det_roi_scale = cli.get<double>("det-roi-scale");
  if (det_roi_scale < 1.0) det_roi_scale = 1.0;
  int port = cli.get<int>("serve");
  const bool serve_mode = port > 0;
  const bool send_enabled = cli.get<int>("send") != 0;
  const double pitch_limit = cli.get<double>("pitch-limit-deg") / 57.3;
  const double yaw_limit = cli.get<double>("yaw-limit-deg") / 57.3;
  const double yaw_jump = cli.get<double>("yaw-jump-deg") / 57.3;
  const double pitch_jump = cli.get<double>("pitch-jump-deg") / 57.3;
  const std::string snap_folder = cli.get<cv::String>("snap-folder");
  if (!tools::mkdirs(snap_folder)) {
    tools::logger()->error("[gimbal_follow] 建快照目录失败: {}", snap_folder);
    return 1;
  }
  int snap_count = 0;
  while (::access(fmt::format("{}/snap_{}.jpg", snap_folder, snap_count + 1).c_str(), F_OK) == 0)
    snap_count++;
  std::string last_snap;

  auto solver_config = auto_aim::solver_config_from_yaml(config_path);
  if (solver_config.R_camera2gimbal == auto_aim::R_CAMERA2GIMBAL_NOMINAL)
    tools::logger()->warn(
      "[gimbal_follow] R_camera2gimbal 还是理想正装占位值——先跑手眼标定并覆盖 "
      "configs/standard3.yaml（角度会系统性偏）");

  std::unique_ptr<auto_aim::YOLO> yolo;
  std::unique_ptr<auto_aim::Solver> solver;
  std::unique_ptr<io::Gimbal> gimbal;
  try {
    yolo.reset(new auto_aim::YOLO(auto_aim::yolo_config_from_yaml(config_path)));
    solver.reset(new auto_aim::Solver(solver_config));
  } catch (const std::exception & e) {
    tools::logger()->error("[gimbal_follow] 载入检测/解算失败: {}", e.what());
    return 1;
  }

  tools::logger()->info(
    "[gimbal_follow] {}（{}）——电控不在线会阻塞等首帧（standard3.yaml first_q_timeout_s=0）",
    send_enabled ? "真发模式：云台会跟随目标动！" : "dry-run 模式：只打印指令不下发",
    send_enabled ? "send=1" : "send=0");
  try {
    gimbal.reset(new io::Gimbal(config_path));
  } catch (const std::exception & e) {
    tools::logger()->error("[gimbal_follow] 电控串口失败: {}", e.what());
    return 1;
  }

  tools::Exiter exiter;
  io::Galaxy camera(camera_config);

  cv::Mat raw, bgr, disp;
  std::chrono::steady_clock::time_point ts;
  for (int i = 0; i < 5; i++) camera.read(raw, ts);  // 预热
  if (raw.empty()) {
    tools::logger()->error("[gimbal_follow] 预热没拿到帧，退出");
    return 1;
  }
  yolo->detect(raw);  // 预热一次推理（首次含 CUDA 初始化）

  const std::string title = "rm_vision gimbal follow";
  tools::ControlServer cs;
  cs.show_auto = false;
  std::atomic<bool> quit{false};
  std::thread server;
  if (serve_mode) {
    server = std::thread(
      tools::serve_control, port, std::ref(cs), std::ref(quit), std::string("gimbal_follow"));
    tools::logger()->info("[gimbal_follow] 浏览器控制台（端口 {}）：Quit 退出 / Save 存快照", port);
  } else {
    tools::logger()->info("[gimbal_follow] 窗口模式；无显示环境时请用 --serve=8082");
  }

  tools::logger()->info(
    "[gimbal_follow] 绘制：{}；检测 ROI：{}",
    draw_enabled ? "开（每帧叠加 + 推流 ~30fps）" : "关（--draw=0：热路径只跑检测/解算/发送）",
    det_roi_enabled ? fmt::format("开（目标框扩边 {:.1f}x，最小边 256px）", det_roi_scale)
                    : "关（全图检测）");
  if (!draw_enabled && serve_mode)
    tools::logger()->info(
      "[gimbal_follow] draw=0：页面图像不刷新（状态数字照常，Quit/Save 可用，Save 落盘时绘一帧）");
  if (!draw_enabled && !serve_mode)
    tools::logger()->warn("[gimbal_follow] draw=0 且无 --serve：没有画面也没有键盘交互，Ctrl-C 退出");

  Display dsp;
  dsp.send_enabled = send_enabled;
  auto do_snap = [&]() {
    if (disp.empty()) return;
    snap_count++;
    last_snap = fmt::format("{}/snap_{}.jpg", snap_folder, snap_count);
    cv::imwrite(last_snap, disp);
    tools::logger()->info("[gimbal_follow] 已存标注快照: {}", last_snap);
  };

  // ---- 指令状态机 ----
  double last_yaw = 0, last_pitch = 0;  // 上一"有效/保持"指令（发出去的或将要发的）
  bool have_last = false;               // 有上一指令（跳变闸与保持的前提）
  int jump_streak = 0;                  // 连续跳变帧数：<3 拒（保持）≥3 接受（真换目标）
  bool controlling = false;             // 已发过 control=1（丢目标时要回 mode=0 收尾）
  auto t_last_ok = std::chrono::steady_clock::now();  // 最近一次有目标（含被拒的）

  // ---- 动态检测 ROI（det-roi=1）：det_roi 为下帧检测窗（整图系）；连丢 3 帧失效回全图 ----
  cv::Rect det_roi;
  bool det_roi_valid = false;
  int det_roi_miss = 0;
  bool snap_pending = false;  // Save 请求：下一帧绘制时落盘（draw=0 时也能存快照）

  bool ever_visible = false;
  uint64_t n = 0, seen_web = 0;
  std::size_t det_count = 0;
  auto t0 = std::chrono::steady_clock::now();
  auto t_last_log = t0, t_last_draw = t0, t_last_status = t0, t_last_snap = t0;
  double fps = 0, fps_avg = 0;

  while (!exiter.exit() && !quit) {
    camera.read(raw, ts);
    if (raw.empty()) continue;
    auto now = std::chrono::steady_clock::now();

    // ---- IMU 同源：q(相机时间戳) → 世界系（对齐同济主链路；链路断会阻塞）----
    Eigen::Quaterniond q = gimbal->q(ts);
    solver->set_R_gimbal2world(q);
    Eigen::Vector3d gimbal_ypr = tools::eulers(solver->R_gimbal2world(), 2, 1, 0);
    double gmb_yaw_world = gimbal_ypr[0];

    // ---- 动态 ROI：上帧目标框扩边 → 本帧只检窗内；窗内没检出同帧回退全图（不漏板）----
    cv::Rect roi_use;
    if (det_roi_enabled && det_roi_valid) roi_use = det_roi;
    std::list<auto_aim::Armor> armors;
    try {
      armors = yolo->detect(raw, roi_use);
      if (!roi_use.empty() && armors.empty()) armors = yolo->detect(raw);
    } catch (const std::exception & e) {
      tools::logger()->error("[gimbal_follow] 检测失败: {}", e.what());
      break;
    }
    const std::string roi_used =
      roi_use.empty() ? "roi full" : fmt::format("roi {}x{}", roi_use.width, roi_use.height);
    det_count = armors.size();
    for (auto & a : armors) solver->solve(a);

    // ---- 选目标：已解算板中"相对云台当前朝向 yaw 偏差"最小者（正跟视野中心）----
    auto_aim::Armor * tgt = nullptr;
    double best = 1e9;
    for (auto & a : armors) {
      if (!a.solved) continue;
      double dy = std::abs(tools::limit_rad(
        std::atan2(a.xyz_in_world.y(), a.xyz_in_world.x()) - gmb_yaw_world));
      if (dy < best) {
        best = dy;
        tgt = &a;
      }
    }

    // ---- 更新下帧检测窗：贴目标扩边；丢目标逐步扩大找，连丢 3 帧回全图 ----
    if (det_roi_enabled) {
      if (tgt) {
        det_roi = expand_det_roi(raw, cv::boundingRect(tgt->points), det_roi_scale);
        det_roi_valid = true;
        det_roi_miss = 0;
      } else if (det_roi_valid) {
        if (++det_roi_miss >= 3) {
          det_roi_valid = false;
        } else {
          det_roi = expand_det_roi(raw, det_roi, 1.5);  // 扩窗速找（兼顾板快速移动）
        }
      }
    }

    // ---- 指令 + 校验 ----
    bool send_now = false;
    double out_yaw = last_yaw, out_pitch = last_pitch, tgt_d = -1;
    std::string state = "IDLE";
    if (tgt) {
      const Eigen::Vector3d & x = tgt->xyz_in_world;
      tgt_d = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      double yaw = tools::limit_rad(std::atan2(x[1], x[0]));
      double pitch = -std::atan2(x[2], std::sqrt(x[0] * x[0] + x[1] * x[1]));  // 向上为负
      t_last_ok = now;

      if (!std::isfinite(yaw) || !std::isfinite(pitch)) {
        dsp.n_nan++;
        state = "DROP-nan";
      } else if (std::abs(pitch) > pitch_limit || std::abs(yaw) > yaw_limit) {
        dsp.n_range++;
        state = "DROP-range";
      } else if (
        have_last && (std::abs(tools::limit_rad(yaw - last_yaw)) > yaw_jump ||
                      std::abs(pitch - last_pitch) > pitch_jump) &&
        ++jump_streak < 3) {
        dsp.n_jump++;
        state = "HOLD-jump";
      } else {
        // 接受：更新 last 并发新值（jump_streak≥3 的"连续跳变"也走这里：真换目标）
        last_yaw = out_yaw = yaw;
        last_pitch = out_pitch = pitch;
        have_last = true;
        jump_streak = 0;
        send_now = true;
        state = send_enabled ? "SEND" : "DRY";
      }
      if (!send_now && have_last) {  // 被拒但有过指令：保持旧值（云台不甩）
        out_yaw = last_yaw;
        out_pitch = last_pitch;
        dsp.n_hold++;
      }
    } else {
      // 无目标：丢 >0.5s 回 mode=0 放弃控制（收尾行为由电控定）；重捕获从头校验
      if (controlling && tools::delta_time(now, t_last_ok) > 0.5) {
        if (send_enabled) gimbal->send(false, false, 0, 0, 0, 0, 0, 0);
        tools::logger()->info("[gimbal_follow] 丢目标 >0.5s：发 mode=0 放弃控制");
        controlling = false;
        have_last = false;
        jump_streak = 0;
        dsp.n_release++;
        det_roi_valid = false;  // 放弃控制后重新从全图找
        det_roi_miss = 0;
      }
      state = "no-target";
    }

    if (send_now) {
      if (send_enabled) gimbal->send(true, false, out_yaw, 0, 0, out_pitch, 0, 0);
      controlling = true;
      dsp.n_sent++;
    }

    // ---- 状态/显示数据 ----
    dsp.cmd_yaw = deg(out_yaw);
    dsp.cmd_pitch = deg(out_pitch);
    dsp.cmd_d = tgt_d;
    dsp.state = state;
    io::GimbalState gs = gimbal->state();
    dsp.gmb_yaw = deg(gs.yaw);
    dsp.gmb_pitch = deg(gs.pitch);

    n++;
    if (tools::delta_time(now, t0) > 0.3) {
      fps = n / tools::delta_time(now, t0);
      fps_avg = fps;
    }
    dsp.fps = fps;

    // ---- 旁路绘制 + 推流（draw=1 时 ~30fps；draw=0 时全跳过，Save 时按需绘一帧）----
    bool drew = false;
    if ((draw_enabled && tools::delta_time(now, t_last_draw) >= 0.033) || snap_pending) {
      cv::cvtColor(raw, bgr, cv::COLOR_BayerBG2BGR);
      if (scale != 1.0)
        cv::resize(bgr, disp, {}, scale, scale, cv::INTER_AREA);
      else
        disp = bgr.clone();  // clone：叠加画在副本上（别学 capture 早期版本污染原图）
      draw_armors(disp, armors, tgt, scale);
      dsp.snap_note = tools::delta_time(now, t_last_snap) < 1.2 ? "SNAPPED" : "";
      draw_overlay(disp, dsp);
      if (serve_mode) cs.streamer.publish(disp);
      drew = true;
      t_last_draw = now;
      if (snap_pending) {  // Save 请求：这一帧画好立刻落盘（draw=0 时也只有这一帧有开销）
        do_snap();
        t_last_snap = now;
        snap_pending = false;
      }
    }

    uint64_t r2 = cs.save_requests.load();
    if (r2 > seen_web) {
      seen_web = r2;
      snap_pending = true;  // 下一帧绘制时落盘（保证快照是全新一帧）
    }

    if (!serve_mode && drew) {
      try {
        cv::imshow(title, disp);
        int k = cv::waitKey(1);
        if (k == 'q' || k == 27) break;
        if (k == 's') snap_pending = true;  // 下一帧绘制时落盘
        if (k == 'w') camera.white_balance_once();
        if (cv::getWindowProperty(title, cv::WND_PROP_VISIBLE) >= 1)
          ever_visible = true;
        else if (ever_visible)
          break;
      } catch (const cv::Exception & e) {
        tools::logger()->error("[gimbal_follow] 打不开窗口（{}）——无显示环境请用 --serve=8082", e.what());
        break;
      }
    }

    if (tools::delta_time(now, t_last_status) >= 0.5) {
      cs.set_status(fmt::format(
        "{{\"fps\":{:.1f},\"det\":{},\"send\":{},\"state\":\"{}\",\"cmd_yaw\":{:.2f},"
        "\"cmd_pitch\":{:.2f},\"gmb_yaw\":{:.2f},\"gmb_pitch\":{:.2f},\"d\":{:.2f},"
        "\"sent\":{},\"hold\":{},\"jump\":{},\"release\":{}}}",
        fps, det_count, send_enabled ? 1 : 0, state, dsp.cmd_yaw, dsp.cmd_pitch, dsp.gmb_yaw,
        dsp.gmb_pitch, tgt_d, dsp.n_sent, dsp.n_hold, dsp.n_jump, dsp.n_release));
      t_last_status = now;
    }

    if (tools::delta_time(now, t_last_log) >= 1.0) {
      double cerr_y = dsp.cmd_yaw - dsp.gmb_yaw, cerr_p = dsp.cmd_pitch - dsp.gmb_pitch;
      while (cerr_y > 180) cerr_y -= 360;
      while (cerr_y <= -180) cerr_y += 360;
      auto yt = yolo->last_timings();
      tools::logger()->info(
        "[gimbal_follow] {} {} yaw={:+.2f} pitch={:+.2f} | gmb yaw={:+.2f} pitch={:+.2f} "
        "cerr={:+.2f}/{:.2f} | d={:.2f}m | {} | ms {:.1f}/{:.1f}/{:.1f} | sent {} hold {} "
        "drop(nan {} rng {} jmp {}) rel {} | 电控 mode={}",
        send_enabled ? "SEND" : "DRY ", state, dsp.cmd_yaw, dsp.cmd_pitch, dsp.gmb_yaw,
        dsp.gmb_pitch, cerr_y, cerr_p, tgt_d, roi_used, yt.preprocess, yt.inference,
        yt.postprocess, dsp.n_sent, dsp.n_hold, dsp.n_nan, dsp.n_range, dsp.n_jump,
        dsp.n_release, gimbal->str(gimbal->mode()));
      t_last_log = now;
    }
  }

  // ---- 收尾：放弃控制（防退出后云台保持最后指令）+ 汇总 ----
  if (send_enabled && controlling) {
    gimbal->send(false, false, 0, 0, 0, 0, 0, 0);
    tools::logger()->info("[gimbal_follow] 退出前已发 mode=0（放弃控制）");
  }
  tools::logger()->info(
    "[gimbal_follow] 汇总：运行 {:.1f}s 帧 {} | sent {} hold {} drop(nan {} rng {} jmp {}) rel {}",
    tools::delta_time(std::chrono::steady_clock::now(), t0), n, dsp.n_sent, dsp.n_hold, dsp.n_nan,
    dsp.n_range, dsp.n_jump, dsp.n_release);

  quit = true;
  if (server.joinable()) server.join();
  tools::logger()->info("[gimbal_follow] 退出");
  return 0;
}
