// 解算（PnP）离线测试：数学自检 / 合成真值闭环 / 真图解算 / 与参考实现对拍 / 性能
//
// 参数必须用 = 传值（OpenCV 解析器规矩），模式用 --mode= 切：
//   ./build/solver_test --mode=math                    # 数学与坐标系约定自检（不读文件，秒级）
//   ./build/solver_test --mode=synth                   # 合成真值闭环：σ=0 精确门 + 噪声/假设偏差扫描
//   ./build/solver_test --mode=type                    # 大/小装甲型别判定（几何反推）特征与规则扫描
//   ./build/solver_test --mode=proj                    # 投影快路径等价性（vs cv::projectPoints 逐值比对）
//   ./build/solver_test --mode=image 图                # 真图：0526 检测 + 逐块解算，打印位姿/耗时
//   ./build/solver_test --mode=image --save=out.png 图 # 叠加图：绿=检出点(标 0..3) 红=重投影点
//   ./build/solver_test --mode=bench                   # 解算耗时 p50/p99（默认 2000 次）
//   ./build/solver_test --mode=xcheck --dump=pts.csv 图# 导出检测点 CSV（给 /tmp 参考实现吃）
//   ./build/solver_test --mode=xcheck --points=pts.csv # 读点 CSV 解算，%.17g 打印（两边 diff）
//
// 合成闭环原理：先定真值（世界系位置 xyz + 装甲板 yaw + pitch 假设），用同一套
// 相机模型正向投影出 4 个像素点（可选加噪），当作"检测结果"喂给 solver，再比对
// 解出来的位姿与真值。σ=0 时全链路精确可逆——不过就是点序/坐标系/欧拉角有错。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace
{
constexpr double DEG = CV_PI / 180.0;

int g_checks = 0, g_fails = 0;

void check(bool ok, const std::string & what)
{
  g_checks++;
  if (!ok) g_fails++;
  std::printf("  [%s] %s\n", ok ? "ok" : "!!", what.c_str());
}

double ms_since(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double pct(std::vector<double> v, double p)
{
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[(std::size_t)(p * (v.size() - 1))];
}

double mean(const std::vector<double> & v)
{
  if (v.empty()) return 0;
  double s = 0;
  for (double x : v) s += x;
  return s / v.size();
}

/// 两条 yaw 的最短角差（度）
double yaw_err_deg(double a, double b)
{
  return std::abs(tools::limit_rad(a - b)) * 180.0 / CV_PI;
}

/// 合成用装甲板：直接指定型别/名称（不经过 ctor 的 ratio 判定——这一步测的是 solver）
auto_aim::Armor make_armor(
  const std::vector<cv::Point2f> & pts, auto_aim::ArmorType type, auto_aim::ArmorName name)
{
  auto_aim::Armor armor(1, 1, 1.0f, cv::boundingRect(pts), pts);
  armor.type = type;
  armor.name = name;
  return armor;
}

/// 重投影残差：解出的位姿（位置 + 精搜索后的 yaw）投回图像，与给定 4 点的最大像素差
double reproj_residual_px(
  const auto_aim::Solver & solver, const auto_aim::Armor & armor)
{
  const auto rp =
    solver.reproject_armor(armor.xyz_in_world, armor.ypr_in_world[0], armor.type, armor.name);
  double m = 0;
  for (int i = 0; i < 4; i++) m = std::max(m, (double)cv::norm(armor.points[i] - rp[i]));
  return m;
}

auto_aim::SolverConfig solver_cfg_from_cli(const cv::CommandLineParser & cli)
{
  auto cfg = auto_aim::solver_config_from_yaml((std::string)cli.get<cv::String>("config"));
  const cv::String center = cli.get<cv::String>("center");
  if (!center.empty()) cfg.yaw_search_center = center;
  const double ap = cli.get<double>("armor_pitch");
  if (ap > -50) cfg.armor_pitch_deg = ap;
  const double op = cli.get<double>("outpost_pitch");
  if (op > -50) cfg.outpost_pitch_deg = op;
  return cfg;
}

// ---------------------------------------------------------------------------
// --mode=math：数学与约定自检
// ---------------------------------------------------------------------------
int run_math()
{
  std::printf("== 数学自检（tools/math_tools 移植）==\n");
  const double PI = CV_PI;

  check(std::abs(tools::limit_rad(0.0)) < 1e-15, "limit_rad(0) = 0");
  check(std::abs(tools::limit_rad(PI - 1e-9) - (PI - 1e-9)) < 1e-12, "limit_rad(π-ε) 不动");
  check(std::abs(tools::limit_rad(PI + 0.1) - (0.1 - PI)) < 1e-12, "limit_rad(π+0.1) = 0.1-π");
  check(std::abs(tools::limit_rad(-PI) - PI) < 1e-12, "limit_rad(-π) = π（区间是 (-π, π]）");
  check(std::abs(tools::limit_rad(-3.5 * PI) - 0.5 * PI) < 1e-12, "limit_rad(-3.5π) = 0.5π");
  check(std::abs(tools::limit_rad(4 * PI + 0.5) - 0.5) < 1e-12, "limit_rad(4π+0.5) = 0.5");

  struct YprCase
  {
    double y, p, r;
  };
  const std::vector<YprCase> cases = {
    {0, 0, 0},        {0.3, -0.2, 0.1},   {-1.2, 0.6, -0.4}, {PI - 0.001, 0.0, 0.0},
    {-PI + 0.001, 0.3, 0.2}, {0.0, 1.5, 0.0}, {2.0, -1.2, 3.0}, {0.0, 0.0, 0.7},
    {1.0, 0.0, -0.9}, {0.0, PI / 2 - 0.02, 0.0}, {0.0, -PI / 2 + 0.02, 0.0}};
  double max_mat = 0, max_ang = 0;
  for (const auto & c : cases) {
    const Eigen::Vector3d ypr(c.y, c.p, c.r);
    const Eigen::Matrix3d R = tools::rotation_matrix(ypr);
    const Eigen::Vector3d back = tools::eulers(R, 2, 1, 0);
    max_mat = std::max(max_mat, (tools::rotation_matrix(back) - R).cwiseAbs().maxCoeff());
    max_ang = std::max(max_ang, (back - ypr).cwiseAbs().maxCoeff());
  }
  char buf[256];
  std::snprintf(buf, sizeof buf, "eulers∘rotation_matrix 矩阵往返 max err = %.2e", max_mat);
  check(max_mat < 1e-12, buf);
  std::snprintf(
    buf, sizeof buf, "eulers∘rotation_matrix 角度往返 max err = %.2e (%.1e°)", max_ang,
    max_ang * 180.0 / PI);
  check(max_ang < 1e-9, buf);

  const Eigen::Vector3d ypd = tools::xyz2ypd(Eigen::Vector3d(1, 1, 1));
  check(
    std::abs(ypd[0] - PI / 4) < 1e-12 && std::abs(ypd[1] - std::atan2(1.0, std::sqrt(2.0))) < 1e-15 &&
      std::abs(ypd[2] - std::sqrt(3.0)) < 1e-15,
    "xyz2ypd(1,1,1) = (45°, 35.2644°, √3)");

  // ---- 坐标系约定自检：合成 K + 名义外参，正前方 3m 一块正对小装甲板 ----
  std::printf("\n== 坐标系约定自检（合成 K，名义外参）==\n");
  auto_aim::SolverConfig cfg;
  cfg.camera_matrix = {1000, 0, 640, 0, 1000, 512, 0, 0, 1};
  cfg.distort_coeffs = {0, 0, 0, 0, 0};
  cfg.armor_pitch_deg = 0;  // 正对（竖立）板
  cfg.yaw_search_center = "pnp";
  auto_aim::Solver solver(cfg);

  const auto pts = solver.reproject_armor(
    Eigen::Vector3d(3, 0, 0), 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::one);
  const cv::Point2f c = (pts[0] + pts[1] + pts[2] + pts[3]) / 4;
  check(pts[1].x > pts[0].x, "TR 在 TL 的 +x（图像右）侧");
  check(pts[2].y > pts[1].y, "BR 在 TR 的 +y（图像下）侧");
  check(
    std::abs(c.x - 640) < 1.0 && std::abs(c.y - 512) < 1.0, "正前方 3m 处的板投影在光心附近");
  const double w_px = cv::norm(pts[1] - pts[0]);
  const double h_px = cv::norm(pts[3] - pts[0]);
  std::snprintf(
    buf, sizeof buf, "小装甲板宽投影 %.2fpx（应 ≈ fx·0.135/3 = %.2f）", w_px, 1000 * 0.135 / 3);
  check(std::abs(w_px - 1000 * 0.135 / 3) < 1.0, buf);
  std::snprintf(
    buf, sizeof buf, "装甲板高投影 %.2fpx（应 ≈ fx·0.056/3 = %.2f）", h_px, 1000 * 0.056 / 3);
  check(std::abs(h_px - 1000 * 0.056 / 3) < 1.0, buf);

  const auto pts_left = solver.reproject_armor(
    Eigen::Vector3d(3, 0.3, 0), 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::one);
  const auto pts_up = solver.reproject_armor(
    Eigen::Vector3d(3, 0, 0.3), 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::one);
  check((pts_left[0] + pts_left[2]).x / 2 < 640, "世界 +y 在图像左侧（y 左约定）");
  check((pts_up[0] + pts_up[2]).y / 2 < 512, "世界 +z 在图像上方（z 上约定）");

  std::printf("\n%s：%d 项检查，%d 项失败\n", g_fails ? "不通过" : "全部通过", g_checks, g_fails);
  return g_fails ? 1 : 0;
}

// ---------------------------------------------------------------------------
// --mode=synth：合成真值闭环
// ---------------------------------------------------------------------------
struct SynthCase
{
  double d, bearing_deg, h, yaw_deg;
  auto_aim::ArmorType type;
  auto_aim::ArmorName name;
  const char * tag;
};

std::vector<SynthCase> sweep_a_cases()
{
  using auto_aim::ArmorName;
  using auto_aim::ArmorType;
  std::vector<SynthCase> v;
  for (double d : {1.0, 2.0, 3.0, 5.0, 8.0})
    for (double b : {-20.0, 0.0, 20.0})
      for (double yaw : {-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0})
        v.push_back({d, b, 0.0, yaw, ArmorType::small, ArmorName::one, "small/one"});
  v.push_back({3.0, 0.0, 0.5, 30.0, ArmorType::small, ArmorName::one, "抬高 0.5m"});
  v.push_back({3.0, 0.0, -0.3, -30.0, ArmorType::small, ArmorName::one, "压低 0.3m"});
  v.push_back({2.0, 10.0, 0.2, 0.0, ArmorType::big, ArmorName::three, "平衡三板(跳过搜索)"});
  v.push_back({4.0, -10.0, 0.0, 45.0, ArmorType::big, ArmorName::one, "大型非平衡"});
  v.push_back({5.0, 0.0, 0.0, 0.0, ArmorType::small, ArmorName::outpost, "前哨(pitch -15)"});
  return v;
}

/// σ=0 精确门：全链路应精确可逆，返回失败数
int sweep_a(const auto_aim::Solver & solver, const auto_aim::Solver & gen)
{
  std::printf("== Sweep A：σ=0 精确门（距离 ≤1mm、yaw ≤0.6°、重投影 ≤0.01px）==\n");
  double max_pos = 0, max_yaw = 0, max_rep = 0, max_yaw_raw = 0;
  int n = 0, fail = 0;
  for (const auto & c : sweep_a_cases()) {
    const double br = c.bearing_deg * DEG;
    const Eigen::Vector3d xyz_true(c.d * std::cos(br), c.d * std::sin(br), c.h);
    const double yaw_true = c.yaw_deg * DEG;
    const auto pts = gen.reproject_armor(xyz_true, yaw_true, c.type, c.name);
    auto armor = make_armor(pts, c.type, c.name);
    n++;
    if (!solver.solve(armor)) {
      std::printf("  [!!] %-18s d=%.0f 方位=%+.0f° 高度=%+.1f yaw=%+.0f° → solve 失败\n",
        c.tag, c.d, c.bearing_deg, c.h, c.yaw_deg);
      fail++;
      continue;
    }
    const double pos_mm = (armor.xyz_in_world - xyz_true).norm() * 1000.0;
    const double yaw_e = yaw_err_deg(armor.ypr_in_world[0], yaw_true);
    const double yaw_raw_e = yaw_err_deg(armor.yaw_raw, yaw_true);
    const double rep = reproj_residual_px(solver, armor);
    max_pos = std::max(max_pos, pos_mm);
    max_yaw = std::max(max_yaw, yaw_e);
    max_yaw_raw = std::max(max_yaw_raw, yaw_raw_e);
    max_rep = std::max(max_rep, rep);
    if (pos_mm > 1.0 || yaw_e > 0.6 || rep > 0.01) {
      std::printf(
        "  [!!] %-18s d=%.0f 方位=%+.0f° 高度=%+.1f yaw=%+.0f° → 位置 %.3fmm yaw %.4f° "
        "重投影 %.4fpx\n",
        c.tag, c.d, c.bearing_deg, c.h, c.yaw_deg, pos_mm, yaw_e, rep);
      fail++;
    }
  }
  std::printf(
    "  %d 例：位置 max %.4f mm | 精搜索后 yaw max %.4f° | PnP 原始 yaw max %.4f° | "
    "重投影 max %.5f px\n",
    n, max_pos, max_yaw, max_yaw_raw, max_rep);
  std::printf("  %s\n", fail ? "有超差 → 移植有错，必须查" : "全绿 ✓");
  return fail;
}

/// 加噪解算一组，回填误差
struct NoiseStat
{
  int n = 0, ok = 0;
  std::vector<double> pos_mm, rel_pct, yaw_raw_deg, yaw_final_deg;
};

NoiseStat sweep_noise_group(
  const auto_aim::Solver & solver, const auto_aim::Solver & gen, const Eigen::Vector3d & xyz_true,
  double yaw_true, auto_aim::ArmorType type, auto_aim::ArmorName name, double sigma, int trials,
  std::mt19937 & rng)
{
  std::normal_distribution<double> nd(0.0, sigma);
  NoiseStat st;
  for (int t = 0; t < trials; t++) {
    auto pts = gen.reproject_armor(xyz_true, yaw_true, type, name);
    for (auto & p : pts) {
      p.x += (float)nd(rng);
      p.y += (float)nd(rng);
    }
    auto armor = make_armor(pts, type, name);
    st.n++;
    if (!solver.solve(armor)) continue;
    st.ok++;
    st.pos_mm.push_back((armor.xyz_in_world - xyz_true).norm() * 1000.0);
    st.rel_pct.push_back(
      std::abs(armor.xyz_in_world.norm() - xyz_true.norm()) / xyz_true.norm() * 100.0);
    st.yaw_raw_deg.push_back(yaw_err_deg(armor.yaw_raw, yaw_true));
    st.yaw_final_deg.push_back(yaw_err_deg(armor.ypr_in_world[0], yaw_true));
  }
  return st;
}

void print_noise_row(const char * label, const NoiseStat & st)
{
  std::printf(
    "  %-16s %6.1f %6.1f   %6.3f %6.3f   %6.3f %6.3f   %6.3f %6.3f   %d/%d\n", label,
    mean(st.pos_mm), pct(st.pos_mm, 0.95), mean(st.rel_pct), pct(st.rel_pct, 0.95),
    mean(st.yaw_raw_deg), pct(st.yaw_raw_deg, 0.95), mean(st.yaw_final_deg),
    pct(st.yaw_final_deg, 0.95), st.ok, st.n);
}

int run_synth(const cv::CommandLineParser & cli)
{
  auto cfg = solver_cfg_from_cli(cli);
  const double sigma = cli.get<double>("sigma") > 0 ? cli.get<double>("sigma") : 1.0;
  const int trials = cli.get<int>("trials") > 0 ? cli.get<int>("trials") : 200;

  auto_aim::Solver solver(cfg);
  auto_aim::Solver gen(cfg);  // 生成器与解算器同配置（Sweep C 里再分别改 pitch）
  std::printf(
    "== 合成真值闭环 ==  配置 %s  噪声 σ=%.2fpx  每组 %d 次\n",
    ((std::string)cli.get<cv::String>("config")).c_str(), sigma, trials);

  int fails = sweep_a(solver, gen);

  // ---- Sweep B：σ=1px 主线（真实噪声下的精度与成功率）----
  std::printf(
    "\n== Sweep B：σ=%.2fpx 噪声（每组 %d 次；距离相对误差 = |d_est-d_true|/d_true）==\n", sigma,
    trials);
  std::printf(
    "  %-16s %6s %6s   %6s %6s   %7s %7s   %7s %7s   %s\n", "组", "位置均", "位置95", "距离均%",
    "距离95%", "raw均°", "raw95°", "终均°", "终95°", "成功");
  std::mt19937 rng(42);
  for (double d : {2.0, 3.0, 5.0}) {
    for (double yaw : {0.0, 30.0}) {
      const Eigen::Vector3d xyz_true(d, 0, 0);
      const auto st = sweep_noise_group(
        solver, gen, xyz_true, yaw * DEG, auto_aim::ArmorType::small, auto_aim::ArmorName::one,
        sigma, trials, rng);
      char label[32];
      std::snprintf(label, sizeof label, "%.0fm yaw%+.0f°", d, yaw);
      print_noise_row(label, st);
    }
  }
  std::printf(
    "  说明：终 yaw 含 1° 步长搜索的量化误差（≤0.5°）；单目测距误差 ~1/板宽像素数，"
    "3m 处 1~2%% 属正常\n");

  // ---- Sweep C：pitch 假设偏差（桌面竖立打印板 vs 规则 15° 倾角）----
  // 注意要带偏离光轴的方位角：在光轴上时 pitch 误差主要把角点往深度里挪（径向），
  // 几乎不影响 yaw 拟合——偏差量随偏离光轴增大
  std::printf("\n== Sweep C：pitch 假设偏差（σ=0，纯系统偏差；真 pitch ≠ 假设 pitch）==\n");
  std::printf(
    "  %-16s %6s %10s %12s %12s\n", "真 pitch → 假设 pitch", "方位", "位置 mm", "raw yaw°",
    "终 yaw°");
  for (double bearing_deg : {0.0, 20.0}) {
    for (double true_pitch : {15.0, 0.0, -15.0}) {
      for (double assumed : {15.0, 0.0}) {
        auto cfg_gen = cfg;
        cfg_gen.armor_pitch_deg = true_pitch;
        auto_aim::Solver gen_p(cfg_gen);
        auto cfg_sv = cfg;
        cfg_sv.armor_pitch_deg = assumed;
        auto_aim::Solver sv_p(cfg_sv);
        const double br = bearing_deg * DEG;
        const Eigen::Vector3d xyz_true(3 * std::cos(br), 3 * std::sin(br), 0);
        const auto pts = gen_p.reproject_armor(
          xyz_true, 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::one);
        auto armor = make_armor(pts, auto_aim::ArmorType::small, auto_aim::ArmorName::one);
        char label[32];
        std::snprintf(label, sizeof label, "%+.0f° → %+.0f°", true_pitch, assumed);
        if (!sv_p.solve(armor)) {
          std::printf("  %-16s %5.0f° 解算失败\n", label, bearing_deg);
          continue;
        }
        std::printf(
          "  %-16s %5.0f° %10.3f %12.4f %12.4f\n", label, bearing_deg,
          (armor.xyz_in_world - xyz_true).norm() * 1000.0, yaw_err_deg(armor.yaw_raw, 0.0),
          yaw_err_deg(armor.ypr_in_world[0], 0.0));
      }
    }
  }
  std::printf(
    "  说明：位置/PnP yaw 不吃 pitch 假设（solvePnP 不知道它），只影响精搜索后的 yaw；"
    "桌面竖板用 --armor_pitch=0\n");

  // ---- Sweep D：yaw 搜索中心与边界（决策 2 的证据）----
  std::printf(
    "\n== Sweep D：搜索中心/边界（auto/pnp/gimbal/bearing；圈 17 个真 yaw，目标在光轴上）==\n");
  for (const char * center : {"auto", "pnp", "gimbal", "bearing"}) {
    auto cfg_c = cfg;
    cfg_c.yaw_search_center = center;
    auto_aim::Solver sv(cfg_c);
    std::printf("  --center=%s\n", center);
    std::printf(
      "    %-9s %11s %11s %11s %9s %11s %7s\n", "真 yaw", "σ=0 终 yaw°", "σ=1 均值°",
      "σ=1 95%°", "σ=1 翻转", "raw 均°", "落窗沿");
    int trunc = 0, flip = 0, ncase = 0;
    for (double yaw_true = -80.0; yaw_true <= 80.1; yaw_true += 10.0) {
      const Eigen::Vector3d xyz_true(3, 0, 0);
      const auto pts = gen.reproject_armor(
        xyz_true, yaw_true * DEG, auto_aim::ArmorType::small, auto_aim::ArmorName::one);
      auto a0 = make_armor(pts, auto_aim::ArmorType::small, auto_aim::ArmorName::one);
      double e0 = -1, edge_dist = 999;
      if (sv.solve(a0)) {
        e0 = yaw_err_deg(a0.ypr_in_world[0], yaw_true * DEG);
        // 解的 yaw 离搜索窗中心多远——>69.5° 说明顶在 ±70° 窗沿上
        const double center_yaw = (std::string(center) == "pnp") ? yaw_true * DEG : 0.0;
        edge_dist = std::abs(tools::limit_rad(a0.ypr_in_world[0] - center_yaw)) * 180.0 / CV_PI;
      }
      std::mt19937 rng(7);
      const auto st = sweep_noise_group(
        sv, gen, xyz_true, yaw_true * DEG, auto_aim::ArmorType::small, auto_aim::ArmorName::one,
        sigma, 30, rng);
      int flips = 0;
      for (double e : st.yaw_final_deg)
        if (e > 90.0) flips++;
      std::printf(
        "    %+-9.0f %11.4f %11.3f %11.3f %9d %11.4f %7s\n", yaw_true, e0,
        mean(st.yaw_final_deg), pct(st.yaw_final_deg, 0.95), flips, mean(st.yaw_raw_deg),
        edge_dist >= 69.5 ? "是" : "");
      ncase++;
      if (std::abs(yaw_true) > 70.0 && e0 > 0.6) trunc++;
      flip += flips;
    }
    std::printf(
      "    => %d 组：截断（|真 yaw|>70° 且 σ=0 误差>0.6°）%d 组，σ=%.2f 下翻转 %d 次\n", ncase,
      trunc, sigma, flip);
  }

  // ---- Sweep E：噪声敏感曲线（给 tracker 的 R 初值定调）----
  std::printf("\n== Sweep E：噪声敏感曲线（3m / yaw=0 / 每组 %d 次）==\n", trials);
  std::printf("  %-8s %8s %8s   %8s %8s   %8s %8s\n", "σ(px)", "位置均", "位置95", "距离均%",
    "距离95%", "raw均°", "终均°");
  for (double s : {0.0, 0.5, 1.0, 2.0}) {
    std::mt19937 rng(123);
    const auto st = sweep_noise_group(
      solver, gen, Eigen::Vector3d(3, 0, 0), 0.0, auto_aim::ArmorType::small,
      auto_aim::ArmorName::one, s, trials, rng);
    char label[16];
    std::snprintf(label, sizeof label, "%.1f", s);
    std::printf(
      "  %-8s %8.1f %8.1f   %8.3f %8.3f   %8.3f %8.3f\n", label, mean(st.pos_mm),
      pct(st.pos_mm, 0.95), mean(st.rel_pct), pct(st.rel_pct, 0.95), mean(st.yaw_raw_deg),
      mean(st.yaw_final_deg));
  }

  std::printf("\n%s：Sweep A %d 项超差\n", fails ? "不通过" : "通过（A 门全绿）", fails);
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------
// --mode=type：大/小装甲型别判定（几何反推）扫描
// ---------------------------------------------------------------------------
// 背景：0526 模型只给编号；而三/四/五号板小号与大号都存在（参考实现
// armor_properties 里 three/four/five 各有 small 与 big 两套类）→ 大小只能从
// 关键点几何反推。对比三条规则（σ=0 系统行为 + σ>0 噪声鲁棒性）：
//   旧：max(上下边)/max(左右边) ≥ 1.4
//   阈值：max(上下边)/两灯条平均长 ≥ big_ratio_threshold（yaml，3.3）
//   现：阈值 + 与型别无关的斜视折算（门槛常量在 armor.cpp，本程序调真实现）
// 另给"正确型别 vs 错误型别"的整链残差差，判断解算侧能否用残差反推型别。

double median_nonneg(std::vector<double> v)
{
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return x < 0; }), v.end());
  return v.empty() ? -1.0 : pct(v, 0.5);
}

int run_type(const cv::CommandLineParser & cli)
{
  auto cfg = solver_cfg_from_cli(cli);
  const double fx = cfg.camera_matrix[0];
  const double thr =
    auto_aim::yolo_config_from_yaml((std::string)cli.get<cv::String>("config")).big_ratio_threshold;
  auto_aim::Solver gen(cfg), solver(cfg);
  using auto_aim::ArmorName;
  using auto_aim::ArmorType;

  std::printf(
    "== 型别判定扫描 ==  合成真值  fx=%.1f  pitch 假设 %.0f°  阈值 %.2f\n", fx,
    cfg.armor_pitch_deg, thr);

  // 解一块板、返回最大重投影残差 px（失败返回 -1）
  auto solve_res = [&](const std::vector<cv::Point2f> & p, ArmorType t) {
    auto a = make_armor(p, t, ArmorName::one);
    if (!solver.solve(a)) return -1.0;
    return reproj_residual_px(solver, a);
  };
  auto xyz_of = [](double d, double bearing_deg) {
    const double br = bearing_deg * DEG;
    return Eigen::Vector3d(d * std::cos(br), d * std::sin(br), 0);
  };

  // ---- [1] σ=0 系统性：每条件扫 yaw −80..80 步长 10，方位 0/20 ----
  std::printf("\n[1] σ=0 系统性（yaw −80..80° 步长 10°；误判数=与真值不符的 yaw 数）\n");
  std::printf(
    "  %-18s %-16s %-15s %s\n", "组", "r_mean 范围", "Δh 范围(px)", "误判 旧/阈值/现（共 n） | 折算开");
  int tot_old = 0, tot_thr = 0, tot_now = 0, tot_n = 0;
  for (auto type : {ArmorType::small, ArmorType::big}) {
    const bool truth_big = (type == ArmorType::big);
    for (double d : {1.0, 2.0, 3.0, 5.0}) {
      for (double bearing_deg : {0.0, 20.0}) {
        int n = 0, e_old = 0, e_thr = 0, e_now = 0, gopen = 0;
        double rmin = 1e9, rmax = -1e9, dhmin = 1e9, dhmax = -1e9;
        std::vector<std::string> err_rows;
        for (double yaw = -80.0; yaw <= 80.1; yaw += 10.0) {
          const auto pts =
            gen.reproject_armor(xyz_of(d, bearing_deg), yaw * DEG, type, ArmorName::one);
          const auto f = auto_aim::armor_type_features(pts, fx);
          const bool old_big = f.ratio_max >= 1.4;
          const bool thr_big = f.ratio_mean >= thr;
          const bool now_big = auto_aim::infer_armor_type(1, pts, thr, fx) == ArmorType::big;
          n++;
          e_old += (old_big != truth_big);
          e_thr += (thr_big != truth_big);
          e_now += (now_big != truth_big);
          gopen += f.oblique_ok;
          rmin = std::min(rmin, f.ratio_mean);
          rmax = std::max(rmax, f.ratio_mean);
          dhmin = std::min(dhmin, f.dh_px);
          dhmax = std::max(dhmax, f.dh_px);
          if (now_big != truth_big)  // 只列现规则的错例（旧规则的错例太多且已定案）
            err_rows.push_back(cv::format(
              "        yaw=%+-4.0f r_max=%.2f r_mean=%.2f Δh=%.2fpx ψ估=%+.1f° 折算=%s→%.2f  "
              "旧=%s 阈=%s 现=%s",
              yaw, f.ratio_max, f.ratio_mean, f.dh_px, f.psi_deg, f.oblique_ok ? "开" : "关",
              f.ratio_used, old_big ? "大" : "小", thr_big ? "大" : "小", now_big ? "大" : "小"));
        }
        std::printf(
          "  %-18s [%.2f, %.2f]     [%.2f, %.2f]     %d/%d/%d（共 %d） | %d\n",
          cv::format("%s %.0fm 方位%.0f°", auto_aim::ARMOR_TYPES[type].c_str(), d, bearing_deg)
            .c_str(),
          rmin, rmax, dhmin, dhmax, e_old, e_thr, e_now, n, gopen);
        for (const auto & r : err_rows) std::printf("%s\n", r.c_str());
        tot_old += e_old;
        tot_thr += e_thr;
        tot_now += e_now;
        tot_n += n;
      }
    }
  }
  std::printf("  合计 %d 例：旧 %d 错 / 阈值 %d 错 / 现 %d 错\n", tot_n, tot_old, tot_thr, tot_now);

  // ---- [2] 残差对比：正确型别 vs 错误型别的整链残差（判断解算侧能否反推型别）----
  std::printf("\n[2] 整链残差（正确 vs 另一型别模型；σ=1 为 30 次中位，px）\n");
  std::printf("  %-14s %-5s %-22s %-24s %s\n", "组", "yaw", "σ=0 正确/错误", "σ=1 正确/错误", "残差差(σ1)");
  std::mt19937 rng(99);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (auto type : {ArmorType::small, ArmorType::big}) {
    const ArmorType other = (type == ArmorType::big) ? ArmorType::small : ArmorType::big;
    for (double d : {1.0, 2.0, 3.0, 5.0}) {
      for (double yaw : {0.0, 20.0, 40.0, 60.0}) {
        const auto pts = gen.reproject_armor(xyz_of(d, 0.0), yaw * DEG, type, ArmorName::one);
        const double r_ok0 = solve_res(pts, type), r_bad0 = solve_res(pts, other);
        std::vector<double> ok1, bad1;
        for (int t = 0; t < 30; t++) {
          auto pn = pts;
          for (auto & p : pn) {
            p.x += (float)nd(rng);
            p.y += (float)nd(rng);
          }
          ok1.push_back(solve_res(pn, type));
          bad1.push_back(solve_res(pn, other));
        }
        const double m_ok = median_nonneg(ok1), m_bad = median_nonneg(bad1);
        std::printf(
          "  %-14s %+-4.0f  %7.2f / %7.2f      %7.2f / %7.2f      %+.2f\n",
          cv::format("%s %.0fm", auto_aim::ARMOR_TYPES[type].c_str(), d).c_str(), yaw, r_ok0,
          r_bad0, m_ok, m_bad, m_bad - m_ok);
      }
    }
  }
  std::printf("  说明：残差差大 → 解算侧可用\"两模型各解一次取残差小者\"补型别；\n");

  // ---- [3] 噪声鲁棒性：σ>0 时三条规则的误判率（只看特征，不解算）----
  std::printf("\n[3] 噪声鲁棒性（每组 7 个真 yaw × 200 次；yaw −60..60 步长 20）\n");
  std::printf(
    "  %-14s %-7s %-24s %-10s %s\n", "组", "σ(px)", "误判率% 旧/阈值/现", "折算开率%", "现规则误判样本(≤3)");
  for (auto type : {ArmorType::small, ArmorType::big}) {
    const bool truth_big = (type == ArmorType::big);
    for (double d : {1.0, 2.0, 3.0}) {
      for (double sigma : {0.5, 1.0, 1.5}) {
        std::mt19937 rng2(2025);
        std::normal_distribution<double> nd2(0.0, sigma);
        int n = 0, e_old = 0, e_thr = 0, e_now = 0, gopen = 0;
        std::vector<std::string> samples;
        for (double yaw : {-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0}) {
          for (int it = 0; it < 200; it++) {
            auto pts = gen.reproject_armor(xyz_of(d, 0.0), yaw * DEG, type, ArmorName::one);
            for (auto & p : pts) {
              p.x += (float)nd2(rng2);
              p.y += (float)nd2(rng2);
            }
            const auto f = auto_aim::armor_type_features(pts, fx);
            const bool now_big = auto_aim::infer_armor_type(1, pts, thr, fx) == ArmorType::big;
            n++;
            e_old += ((f.ratio_max >= 1.4) != truth_big);
            e_thr += ((f.ratio_mean >= thr) != truth_big);
            e_now += (now_big != truth_big);
            gopen += f.oblique_ok;
            if (now_big != truth_big && samples.size() < 3)
              samples.push_back(cv::format(
                "yaw%.0f r_mean=%.2f Δh=%.1f ψ估=%+.1f° 折算=%s→%.2f", yaw, f.ratio_mean, f.dh_px,
                f.psi_deg, f.oblique_ok ? "开" : "关", f.ratio_used));
          }
        }
        std::string s;
        for (const auto & x : samples) s += "[" + x + "] ";
        std::printf(
          "  %-14s %-7.1f %5.1f / %5.1f / %5.1f       %5.1f      %s\n",
          cv::format("%s %.0fm", auto_aim::ARMOR_TYPES[type].c_str(), d).c_str(), sigma,
          100.0 * e_old / n, 100.0 * e_thr / n, 100.0 * e_now / n, 100.0 * gopen / n, s.c_str());
      }
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// --mode=proj：投影快路径等价性（project_armor_fast vs cv::projectPoints）
// [1] 稠密网格逐点比对：两条投影的点差应 ≤0.01px（手写展开 vs OpenCV）
// [2] yaw 搜索等价：同一批样本上，两条误差函数的 140 候选 argmin 必须一致
// （快路径的唯一风险是"错了还看起来对"，所以这步必须是逐值对出来的，不看结论看数值）
// ---------------------------------------------------------------------------
int run_proj(const cv::CommandLineParser & cli)
{
  auto cfg = solver_cfg_from_cli(cli);
  auto_aim::Solver solver(cfg);
  using auto_aim::ArmorName;
  using auto_aim::ArmorType;

  std::printf("== 投影快路径等价性 ==\n");

  // ---- [1] 网格逐点比对 ----
  double max_d = 0;
  std::string worst;
  int np = 0;
  for (auto type : {ArmorType::small, ArmorType::big}) {
    for (auto name : {ArmorName::one, ArmorName::outpost, ArmorName::base}) {
      for (double d : {1.0, 2.0, 3.0, 5.0, 8.0}) {
        for (double bearing_deg : {-25.0, 0.0, 20.0}) {
          for (double yaw_deg = -85.0; yaw_deg <= 85.1; yaw_deg += 5.0) {
            const double br = bearing_deg * DEG;
            const Eigen::Vector3d xyz(d * std::cos(br), d * std::sin(br), 0.1);
            const auto a = solver.reproject_armor(xyz, yaw_deg * DEG, type, name);
            const auto b = solver.project_armor_fast(xyz, yaw_deg * DEG, type, name);
            for (int i = 0; i < 4; i++) {
              const double dd = cv::norm(a[i] - b[i]);
              np++;
              if (dd > max_d) {
                max_d = dd;
                worst = cv::format(
                  "%s/%s d=%.1f 方位%+.0f yaw%+.0f 点%d", auto_aim::ARMOR_TYPES[type].c_str(),
                  auto_aim::ARMOR_NAMES[name].c_str(), d, bearing_deg, yaw_deg, i);
              }
            }
          }
        }
      }
    }
  }
  const bool ok_grid = max_d <= 0.01;
  std::printf(
    "[1] 网格 %d 点：max 点差 %.2e px（判据 ≤0.01）%s\n     最差点: %s\n", np, max_d,
    ok_grid ? "  OK" : "  ！超差", worst.empty() ? "-" : worst.c_str());

  // ---- [2] yaw 搜索 argmin 等价 ----
  // 样本：合成一块板（含噪 σ=1），按 optimize_yaw 的真实口径跑 140 候选
  // （搜索中心取 auto 无 IMU 时的 bearing；与 solver.cpp yaw_center_of 同定义）
  constexpr int SEARCH_RANGE = 140;
  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0.0, 1.0);
  int n_case = 0, n_diff = 0, n_tight = 0;
  double max_de = 0;
  double min_margin_seen = 1e9;
  for (auto type : {ArmorType::small, ArmorType::big}) {
    for (double d : {1.0, 2.0, 3.0, 5.0}) {
      for (double bearing_deg : {0.0, 20.0}) {
        for (double yaw_true_deg : {-60.0, -30.0, 0.0, 30.0, 60.0}) {
          const double br = bearing_deg * DEG;
          const Eigen::Vector3d xyz(d * std::cos(br), d * std::sin(br), 0.1);
          auto pts = solver.reproject_armor(xyz, yaw_true_deg * DEG, type, ArmorName::one);
          for (auto & p : pts) p += cv::Point2f((float)noise(rng), (float)noise(rng));

          const double yaw0 = tools::limit_rad(std::atan2(xyz.y(), xyz.x()) - 70.0 * DEG);
          double best_e[2] = {1e18, 1e18}, second_e[2] = {1e18, 1e18};
          int arg[2] = {-1, -1};
          for (int k = 0; k < SEARCH_RANGE; k++) {
            const double yw = tools::limit_rad(yaw0 + k * DEG);
            const auto pa = solver.reproject_armor(xyz, yw, type, ArmorName::one);
            const auto pb = solver.project_armor_fast(xyz, yw, type, ArmorName::one);
            double ea = 0, eb = 0;
            for (int i = 0; i < 4; i++) {
              ea += cv::norm(pts[i] - pa[i]);
              eb += cv::norm(pts[i] - pb[i]);
            }
            max_de = std::max(max_de, std::abs(ea - eb));
            for (int which = 0; which < 2; which++) {
              const double e = which ? eb : ea;
              if (e < best_e[which]) {
                second_e[which] = best_e[which];
                best_e[which] = e;
                arg[which] = k;
              } else if (e < second_e[which]) {
                second_e[which] = e;
              }
            }
          }
          n_case++;
          if (arg[0] != arg[1]) n_diff++;
          const double margin = second_e[0] - best_e[0];
          min_margin_seen = std::min(min_margin_seen, margin);
          if (margin < 0.1) n_tight++;
        }
      }
    }
  }
  const bool ok_search = (n_diff == 0);
  std::printf(
    "[2] 搜索等价：%d 样本 argmin 不一致 %d 个（判据 0）%s | 候选误差最大差 %.2e px"
    " | OpenCV 路径最优-次优最小间隔 %.3f px（<0.1 的有 %d 个）\n",
    n_case, n_diff, ok_search ? "  OK" : "  ！不一致", max_de, min_margin_seen, n_tight);

  check(ok_grid && ok_search, "投影快路径与 cv::projectPoints 等价");
  if (g_fails == 0 && ok_grid && ok_search) return 0;
  return 1;
}

// ---------------------------------------------------------------------------
// --mode=image：真图检测 + 解算
// ---------------------------------------------------------------------------
int run_image(const cv::CommandLineParser & cli)
{
  const std::string image_path = cli.get<cv::String>("@image");
  if (image_path.empty()) {
    std::printf("用法：--mode=image [--save=out.png] 图片路径\n");
    return 1;
  }
  const std::string config_path = cli.get<cv::String>("config");
  auto ycfg = auto_aim::yolo_config_from_yaml(config_path);
  const double conf = cli.get<double>("conf");
  if (conf > 0) ycfg.min_confidence = conf;
  const double nms = cli.get<double>("nms");
  if (nms > 0) ycfg.nms_threshold = nms;
  // 诊断用：对照"场上无大板(全按小) / 按几何判大小"两条路径
  const int big_armor = cli.get<int>("big_armor");
  if (big_armor >= 0) ycfg.has_big_armor = (big_armor > 0);

  auto_aim::YOLO yolo(ycfg);
  const auto sv_cfg = solver_cfg_from_cli(cli);
  auto_aim::Solver solver(sv_cfg);
  const double fx = sv_cfg.camera_matrix[0];

  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    tools::logger()->error("[solver_test] 读不到图片: {}", image_path);
    return 1;
  }

  // 诊断用：--type=big/small 覆盖检出的大小装甲（默认按检出）。用来验证
  // "型别判错 → 解算距离偏 230/135 倍" 这类问题
  const std::string type_override = cli.get<cv::String>("type");
  if (!type_override.empty() && type_override != "big" && type_override != "small") {
    std::printf("--type 只能是 big / small，实得 %s\n", type_override.c_str());
    return 1;
  }
  auto apply_type_override = [&](std::list<auto_aim::Armor> & as) {
    if (type_override.empty()) return;
    for (auto & a : as)
      a.type = (type_override == "big") ? auto_aim::ArmorType::big : auto_aim::ArmorType::small;
  };

  for (int i = 0; i < 3; i++) {
    auto armors = yolo.detect(img);
    apply_type_override(armors);
    for (auto & a : armors) solver.solve(a);
  }

  constexpr int N = 30;
  std::vector<double> det_ms, solve_ms;
  std::list<auto_aim::Armor> armors;
  for (int i = 0; i < N; i++) {
    auto t0 = std::chrono::steady_clock::now();
    armors = yolo.detect(img);
    det_ms.push_back(ms_since(t0));
    apply_type_override(armors);
    double s = 0;
    for (auto & a : armors) {
      solver.solve(a);
      s += solver.last_timings().total();
    }
    solve_ms.push_back(s);
  }

  std::printf(
    "\n输入 %s  %dx%d  检出 %zu 块（预热 3 次后 %d 次分位数）\n"
    "  检测 p50=%.2f p99=%.2f | 解算(逐帧总和) p50=%.3f p99=%.3f ms\n\n",
    image_path.c_str(), img.cols, img.rows, armors.size(), N, pct(det_ms, 0.5), pct(det_ms, 0.99),
    pct(solve_ms, 0.5), pct(solve_ms, 0.99));

  cv::Mat vis = img.clone();
  int idx = 0;
  for (const auto & a : armors) {
    const auto t = solver.last_timings();
    if (!a.solved) {
      std::printf("[%d] %s/%s %s conf=%.3f → 解算失败\n", idx++, auto_aim::COLORS[a.color].c_str(),
        auto_aim::ARMOR_NAMES[a.name].c_str(), auto_aim::ARMOR_TYPES[a.type].c_str(),
        a.confidence);
      continue;
    }
    const auto tf = auto_aim::armor_type_features(a.points, fx);
    std::printf(
      "[%d] %s/%s %s conf=%.3f ratio=%.2f  [型别: 均值比=%.2f 折算=%s→%.2f ψ估=%+.1f° Δh=%.1fpx]\n"
      "    world=(%+.3f, %+.3f, %+.3f) m   球坐标 yaw=%+.2f° pitch=%+.2f° d=%.3f m\n"
      "    yaw_raw=%+.3f° → 精搜索 %+.3f°   ypr_gimbal=(%+.2f°, %+.2f°, %+.2f°)\n",
      idx, auto_aim::COLORS[a.color].c_str(), auto_aim::ARMOR_NAMES[a.name].c_str(),
      auto_aim::ARMOR_TYPES[a.type].c_str(), a.confidence, a.ratio, tf.ratio_mean,
      tf.oblique_ok ? "开" : "关", tf.ratio_used, tf.psi_deg, tf.dh_px, a.xyz_in_world[0],
      a.xyz_in_world[1],
      a.xyz_in_world[2], a.ypd_in_world[0] * 180 / CV_PI, a.ypd_in_world[1] * 180 / CV_PI,
      a.ypd_in_world[2], a.yaw_raw * 180 / CV_PI, a.ypr_in_world[0] * 180 / CV_PI,
      a.ypr_in_gimbal[0] * 180 / CV_PI, a.ypr_in_gimbal[1] * 180 / CV_PI,
      a.ypr_in_gimbal[2] * 180 / CV_PI);

    // 逐点残差：既看 max 也看方向（点序/尺寸/语义不符时方向有特征——如左右点成对向内）
    const auto rp = solver.reproject_armor(a.xyz_in_world, a.ypr_in_world[0], a.type, a.name);
    const auto rp_raw = solver.reproject_armor(a.xyz_in_world, a.yaw_raw, a.type, a.name);
    double max_e = 0, max_raw = 0;
    std::string per_point;
    for (int k = 0; k < 4; k++) {
      const cv::Point2f d = a.points[k] - rp[k];
      max_e = std::max(max_e, (double)cv::norm(d));
      max_raw = std::max(max_raw, (double)cv::norm(a.points[k] - rp_raw[k]));
      per_point += cv::format(" %d(%+.1f,%+.1f)=%.1f", k, d.x, d.y, (double)cv::norm(d));
    }
    std::printf(
      "    重投影残差 max=%.3f px（原始 yaw 下 %.3f px）  耗时 pnp=%.3f 搜索=%.3f ms\n"
      "    逐点像素差（检出−重投影，x 右 / y 下）:%s\n",
      max_e, max_raw, t.pnp, t.yaw_search, per_point.c_str());

    // 叠加：绿=检出点(标索引) 红=重投影点 黄字=距离/yaw（只能用 ASCII——Hershey 渲染不了中文）
    for (int k = 0; k < 4; k++) {
      cv::circle(vis, a.points[k], 3, cv::Scalar(0, 255, 0), -1);
      cv::putText(
        vis, std::to_string(k), a.points[k] + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.45,
        cv::Scalar(0, 255, 0), 1);
      cv::line(vis, a.points[k], a.points[(k + 1) % 4], cv::Scalar(0, 255, 0), 1);
    }
    for (int k = 0; k < 4; k++) cv::circle(vis, rp[k], 3, cv::Scalar(0, 0, 255), -1);
    cv::putText(
      vis,
      cv::format("d=%.2fm yaw=%.1f", a.ypd_in_world[2], a.ypr_in_world[0] * 180 / CV_PI),
      a.center + cv::Point2f(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    idx++;
  }

  const cv::String save = cli.get<cv::String>("save");
  if (!save.empty()) {
    cv::imwrite(save, vis);
    std::printf("\n叠加图已存: %s（绿=检出点[标 0..3 点序] 红=解算位姿重投影 黄=distance/yaw）\n",
      ((std::string)save).c_str());
  }
  return armors.empty() ? 2 : 0;
}

// ---------------------------------------------------------------------------
// --mode=bench：解算耗时
// ---------------------------------------------------------------------------
int run_bench(const cv::CommandLineParser & cli)
{
  auto cfg = solver_cfg_from_cli(cli);
  auto_aim::Solver solver(cfg);
  const int iters = cli.get<int>("iters") > 0 ? cli.get<int>("iters") : 2000;

  // 合成一块 3m 的板（σ=1px），只测解算本身
  auto_aim::SolverConfig cfg_gen = cfg;
  auto_aim::Solver gen(cfg_gen);
  std::mt19937 rng(1);
  std::normal_distribution<double> nd(0.0, 1.0);
  auto pts = gen.reproject_armor(
    Eigen::Vector3d(3, 0, 0), 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::one);
  for (auto & p : pts) {
    p.x += (float)nd(rng);
    p.y += (float)nd(rng);
  }
  const auto armor = make_armor(pts, auto_aim::ArmorType::small, auto_aim::ArmorName::one);

  for (int i = 0; i < 100; i++) {
    auto a = armor;
    solver.solve(a);
  }
  std::vector<double> pnp(iters), search(iters), tot(iters);
  int ok = 0;
  for (int i = 0; i < iters; i++) {
    auto a = armor;
    if (solver.solve(a)) ok++;
    const auto t = solver.last_timings();
    pnp[i] = t.pnp;
    search[i] = t.yaw_search;
    tot[i] = t.total();
  }
  std::printf(
    "== 解算耗时（%d 次，单块装甲板）==\n"
    "  solvePnP+级联 p50=%.3f p99=%.3f | yaw 精搜索 p50=%.3f p99=%.3f | 合计 p50=%.3f p99=%.3f "
    "ms  (成功 %d/%d)\n"
    "  台账判据：单帧总和 p99 ≤0.3ms 保持忠实实现；0.3~1.0ms 排期优化；>1.0ms 先优化\n"
    "  一帧两块装甲板时总和 ×2 ≈ p99 %.3f ms\n",
    iters, pct(pnp, 0.5), pct(pnp, 0.99), pct(search, 0.5), pct(search, 0.99), pct(tot, 0.5),
    pct(tot, 0.99), ok, iters, pct(tot, 0.99) * 2);
  return 0;
}

// ---------------------------------------------------------------------------
// --mode=xcheck：与参考实现对拍用的点 CSV 导出 / 读入
// ---------------------------------------------------------------------------
int dump_points(const cv::CommandLineParser & cli)
{
  const std::string image_path = cli.get<cv::String>("@image");
  const std::string config_path = cli.get<cv::String>("config");
  auto ycfg = auto_aim::yolo_config_from_yaml(config_path);
  const double conf = cli.get<double>("conf");
  if (conf > 0) ycfg.min_confidence = conf;
  auto_aim::YOLO yolo(ycfg);
  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    std::printf("读不到图片: %s\n", image_path.c_str());
    return 1;
  }
  const auto armors = yolo.detect(img);
  const std::string out_path = cli.get<cv::String>("dump");
  FILE * f = std::fopen(out_path.c_str(), "w");
  if (!f) {
    std::printf("写不了 %s\n", out_path.c_str());
    return 1;
  }
  std::fprintf(
    f, "# solver_test dump-points v1: name type color_id num_id conf x1 y1 x2 y2 x3 y3 x4 y4"
       "  （图片 %s，像素坐标与原图同尺度）\n",
    image_path.c_str());
  for (const auto & a : armors) {
    std::fprintf(
      f, "%s %s %d %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
      auto_aim::ARMOR_NAMES[a.name].c_str(), auto_aim::ARMOR_TYPES[a.type].c_str(), a.color_id,
      a.num_id, a.confidence, a.points[0].x, a.points[0].y, a.points[1].x, a.points[1].y,
      a.points[2].x, a.points[2].y, a.points[3].x, a.points[3].y);
  }
  std::fclose(f);
  std::printf("已导出 %zu 块装甲板的点 → %s\n", armors.size(), out_path.c_str());
  return 0;
}

int solve_points(const cv::CommandLineParser & cli)
{
  const std::string in_path = cli.get<cv::String>("points");
  FILE * f = std::fopen(in_path.c_str(), "r");
  if (!f) {
    std::printf("读不了 %s\n", in_path.c_str());
    return 1;
  }
  auto_aim::Solver solver(solver_cfg_from_cli(cli));
  std::printf("# solver_test results v1: 每行一块装甲板（各分量 17 位有效数字）\n");
  char line[1024];
  int i = 0;
  while (std::fgets(line, sizeof line, f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    char name_s[32] = {0}, type_s[32] = {0};
    int color_id = 0, num_id = 0;
    double conf = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0, x3 = 0, y3 = 0, x4 = 0, y4 = 0;
    if (std::sscanf(
          line, "%31s %31s %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf", name_s, type_s, &color_id,
          &num_id, &conf, &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4) < 13) {
      std::printf("跳过解析不了的行: %s", line);
      continue;
    }
    const std::vector<cv::Point2f> pts = {
      {(float)x1, (float)y1}, {(float)x2, (float)y2}, {(float)x3, (float)y3}, {(float)x4, (float)y4}};
    auto_aim::ArmorType type = auto_aim::ArmorType::small;
    auto_aim::ArmorName name = auto_aim::ArmorName::one;
    for (std::size_t k = 0; k < auto_aim::ARMOR_TYPES.size(); k++)
      if (type_s == auto_aim::ARMOR_TYPES[k]) type = (auto_aim::ArmorType)k;
    for (std::size_t k = 0; k < auto_aim::ARMOR_NAMES.size(); k++)
      if (name_s == auto_aim::ARMOR_NAMES[k]) name = (auto_aim::ArmorName)k;
    auto armor = make_armor(pts, type, name);
    const bool ok = solver.solve(armor);
    auto p = [](const Eigen::Vector3d & v) {
      char b[128];
      std::snprintf(b, sizeof b, "%.17g %.17g %.17g", v[0], v[1], v[2]);
      return std::string(b);
    };
    char head[128];
    std::snprintf(head, sizeof head, "i=%d name=%s type=%s ok=%d", i++, name_s, type_s, ok ? 1 : 0);
    if (!ok) {
      std::printf("%s\n", head);
      continue;
    }
    std::printf(
      "%s xyz_g=%s xyz_w=%s ypr_g=%s ypr_w=%s ypd_w=%s yaw_raw=%.17g\n", head, p(armor.xyz_in_gimbal).c_str(),
      p(armor.xyz_in_world).c_str(), p(armor.ypr_in_gimbal).c_str(), p(armor.ypr_in_world).c_str(),
      p(armor.ypd_in_world).c_str(), armor.yaw_raw);
  }
  std::fclose(f);
  return 0;
}

int run_xcheck(const cv::CommandLineParser & cli)
{
  const cv::String dump = cli.get<cv::String>("dump");
  const cv::String points = cli.get<cv::String>("points");
  if (!dump.empty()) return dump_points(cli);
  if (!points.empty()) return solve_points(cli);
  std::printf("xcheck 模式：--dump=pts.csv 图（导出检测点） 或 --points=pts.csv（读点解算）\n");
  return 1;
}

}  // namespace

int main(int argc, char ** argv)
{
  const cv::String keys =
    "{help h usage ? |      | 打印帮助 }"
    "{mode m         | synth | math / synth / type / proj / image / bench / xcheck }"
    "{@image         |      | 输入图片（image / xcheck --dump 用）}"
    "{config         | configs/standard3.yaml | 配置文件（读顶层解算键 + yolo: 块）}"
    "{conf c         | 0    | objectness 阈值（>0 覆盖 yaml）}"
    "{nms n          | 0    | NMS 阈值（>0 覆盖 yaml）}"
    "{center         |      | 覆盖 yaw_search_center：auto/pnp/gimbal }"
    "{armor_pitch    | -100 | 覆盖 armor_pitch_deg（度；默认用 yaml。桌面竖板用 0）}"
    "{outpost_pitch  | -100 | 覆盖 outpost_pitch_deg（度）}"
    "{sigma          | 1.0  | synth：噪点 σ（像素）}"
    "{trials         | 200  | synth：噪声组每组次数}"
    "{iters          | 2000 | bench：解算次数}"
    "{type           |      | image：强制大小装甲（诊断用，空=按检出）。big/small }"
    "{big_armor      | -1   | 覆盖 has_big_armor：0=场上无大板(今年规则) 1=按几何判 空=-1=用 yaml }"
    "{save s         |      | image：叠加图输出路径 }"
    "{dump           |      | xcheck：导出检测点 CSV 路径 }"
    "{points         |      | xcheck：输入点 CSV 路径 }";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const std::string mode = cli.get<cv::String>("mode");
  // 位置参数防呆：必须用 = 传值，漏写时数值会顶掉默认的 config 路径
  const std::string cfg_path = cli.get<cv::String>("config");
  if (cfg_path.find('/') == std::string::npos && cfg_path.find(".yaml") == std::string::npos) {
    tools::logger()->error(
      "[solver_test] 位置参数 \"{}\" 不像配置文件路径。注意：本程序参数必须用 = 传值（如 "
      "--armor_pitch=0），写 \"--armor_pitch 0\" 会把 0 当成配置文件",
      cfg_path);
    return 1;
  }

  if (mode == "math") return run_math();
  if (mode == "synth") return run_synth(cli);
  if (mode == "type") return run_type(cli);
  if (mode == "proj") return run_proj(cli);
  if (mode == "image") return run_image(cli);
  if (mode == "bench") return run_bench(cli);
  if (mode == "xcheck") return run_xcheck(cli);
  tools::logger()->error("[solver_test] 未知模式: {}", mode);
  return 1;
}
