#include "auto_aim/solver.hpp"

#include <chrono>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace auto_aim
{
namespace
{
// 装甲板实物尺寸（同济写死的常量，保持一致；实测打印板尺寸不对时改这里）
constexpr double LIGHTBAR_LENGTH = 56e-3;     // m，灯条长 = 装甲板高
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // m

// 物点：装甲板本体系（板面在 x=0 平面），+y 向图像左、+z 向图像上，
// 与 Armor::points 的 [左上,右上,右下,左下] 一一对应（同济同款）
const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  cv::Point3f(0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2),
  cv::Point3f(0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2),
  cv::Point3f(0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2),
  cv::Point3f(0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2)};
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  cv::Point3f(0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2),
  cv::Point3f(0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2),
  cv::Point3f(0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2),
  cv::Point3f(0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2)};

/// 9 个数（行主序）→ 3x3；Eigen 的 RowMajor 构造+赋值的组合即按行读数
Eigen::Matrix3d mat3_from(const std::vector<double> & v, const char * what)
{
  if (v.size() != 9)
    throw std::runtime_error(
      std::string("解算配置 ") + what + " 需要 9 个数（行主序），实得 " + std::to_string(v.size()));
  return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(v.data());
}

double delta_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
  return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

SolverConfig solver_config_from_yaml(const std::string & config_path)
{
  auto yaml = tools::load(config_path);

  SolverConfig config;
  config.camera_matrix = tools::read<std::vector<double>>(yaml, "camera_matrix");
  config.distort_coeffs = tools::read<std::vector<double>>(yaml, "distort_coeffs");
  if (config.camera_matrix.size() != 9)
    throw std::runtime_error(
      "yaml camera_matrix 需要 9 个数（行主序），实得 " + std::to_string(config.camera_matrix.size()));
  if (config.distort_coeffs.size() != 5)
    throw std::runtime_error(
      "yaml distort_coeffs 需要 5 个数，实得 " + std::to_string(config.distort_coeffs.size()));

  config.R_gimbal2imubody =
    tools::read_or<std::vector<double>>(yaml, "R_gimbal2imubody", config.R_gimbal2imubody);
  config.R_camera2gimbal =
    tools::read_or<std::vector<double>>(yaml, "R_camera2gimbal", config.R_camera2gimbal);
  config.t_camera2gimbal =
    tools::read_or<std::vector<double>>(yaml, "t_camera2gimbal", config.t_camera2gimbal);
  config.yaw_search_center =
    tools::read_or<std::string>(yaml, "yaw_search_center", config.yaw_search_center);
  config.armor_pitch_deg = tools::read_or<double>(yaml, "armor_pitch_deg", config.armor_pitch_deg);
  config.outpost_pitch_deg =
    tools::read_or<double>(yaml, "outpost_pitch_deg", config.outpost_pitch_deg);
  return config;
}

Solver::Solver(const SolverConfig & config)
{
  if (config.camera_matrix.size() != 9 || config.distort_coeffs.size() != 5)
    throw std::runtime_error(
      "SolverConfig 内参维数不对：camera_matrix 需 9 个数（行主序）、distort_coeffs 需 5 个数");

  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(config.camera_matrix.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(config.distort_coeffs.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  R_gimbal2imubody_ = mat3_from(config.R_gimbal2imubody, "R_gimbal2imubody");
  R_camera2gimbal_ = mat3_from(config.R_camera2gimbal, "R_camera2gimbal");
  if (config.t_camera2gimbal.size() != 3)
    throw std::runtime_error(
      "解算配置 t_camera2gimbal 需要 3 个数，实得 " + std::to_string(config.t_camera2gimbal.size()));
  t_camera2gimbal_ = Eigen::Vector3d(config.t_camera2gimbal.data());

  if (
    config.yaw_search_center != "auto" && config.yaw_search_center != "pnp" &&
    config.yaw_search_center != "gimbal" && config.yaw_search_center != "bearing")
    throw std::runtime_error(
      "yaw_search_center 只能是 auto / pnp / gimbal / bearing，实得 " + config.yaw_search_center);
  yaw_search_center_ = config.yaw_search_center;
  armor_pitch_ = config.armor_pitch_deg * CV_PI / 180.0;
  outpost_pitch_ = config.outpost_pitch_deg * CV_PI / 180.0;

  if (config.R_camera2gimbal == R_CAMERA2GIMBAL_NOMINAL) {
    tools::logger()->warn(
      "[solver] 手眼外参还是\"相机理想正装\"占位值（未标定）：gimbal/world 系结果为相对值，"
      "绝对姿态不可信；标定后覆盖 configs 里的 R_camera2gimbal / t_camera2gimbal");
  }

  tools::logger()->info(
    "[solver] 载入完成: fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}  pitch 假设 {:.0f}°/{:.0f}°  "
    "yaw 搜索中心={}",
    camera_matrix_.at<double>(0, 0), camera_matrix_.at<double>(1, 1), camera_matrix_.at<double>(0, 2),
    camera_matrix_.at<double>(1, 2),
    config.armor_pitch_deg, config.outpost_pitch_deg, yaw_search_center_);
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
  world_attitude_set_ = true;
}

bool Solver::solve(Armor & armor) const
{
  using clock = std::chrono::steady_clock;
  if (armor.points.size() != 4) {
    tools::logger()->warn("[solver] 关键点数不是 4（实得 {}），跳过", armor.points.size());
    return false;
  }

  const auto & object_points = (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  // ---- solvePnP（IPPE 内部会去畸变，畸变系数直接传即可，Jetson OpenCV 4.5.4 实测精确）----
  auto t0 = clock::now();
  cv::Vec3d rvec, tvec;
  const bool ok = cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);
  if (!ok) {
    tools::logger()->warn("[solver] solvePnP(IPPE) 失败（点退化？），跳过");
    return false;
  }
  for (int i = 0; i < 3; i++) {
    if (!std::isfinite(tvec[i])) {
      tools::logger()->warn("[solver] solvePnP 解出非有限位置，跳过");
      return false;
    }
  }
  if (tvec[2] <= 0) {
    tools::logger()->warn("[solver] solvePnP 解出相机后方的板（z={:.3f}），跳过", tvec[2]);
    return false;
  }

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  const Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  const Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  // yaw 精搜索前的原始 yaw。同济只在非平衡分支的 optimize_yaw 里赋值（平衡分支是
  // 未初始化内存——本移植补齐）
  armor.yaw_raw = armor.ypr_in_world[0];
  auto t1 = clock::now();

  // 平衡不做 yaw 优化，因为 pitch 假设不成立（同济注释原话）
  const bool is_balance =
    (armor.type == ArmorType::big) &&
    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
     armor.name == ArmorName::five);
  if (!is_balance) optimize_yaw(armor);
  auto t2 = clock::now();

  armor.solved = true;
  timings_ = {delta_ms(t0, t1), delta_ms(t1, t2)};
  return true;
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  const Eigen::Vector3d ypr(yaw, pitch_of(name), 0.0);
  // 装甲板本体系 → world 系：Rz(yaw)·Ry(pitch)，roll=0（与同济手写的展开矩阵逐项相同）
  const Eigen::Matrix3d R_armor2world = tools::rotation_matrix(ypr);

  // world → camera：外参的逆级联（对齐同济）
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  const Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  const Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  const cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  std::vector<cv::Point2f> image_points;
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

void Solver::optimize_yaw(Armor & armor) const
{
  // 把装甲板绕中心旋转、俯仰固定为假设值，在 ±70°（步长 1°）里找重投影误差最小的 yaw
  constexpr double SEARCH_RANGE = 140;  // degree（同济）
  const double yaw0 =
    tools::limit_rad(yaw_center_of(armor) - SEARCH_RANGE / 2 * CV_PI / 180.0);

  double min_error = 1e10;
  double best_yaw = armor.ypr_in_world[0];
  for (int i = 0; i < (int)SEARCH_RANGE; i++) {
    const double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    const double error = armor_reprojection_error(armor, yaw);
    if (error < min_error) {
      min_error = error;
      best_yaw = yaw;
    }
  }

  armor.ypr_in_world[0] = best_yaw;
}

std::vector<cv::Point2f> Solver::project_armor_fast(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  // 与 reproject_armor 同一链路，只把 Rodrigues/eigen2cv/cv::projectPoints 换成手写展开：
  //   R_armor2world = Rz(yaw)·Ry(pitch)，roll=0
  //   R_armor2camera = R_camera2gimbalᵀ · R_gimbal2worldᵀ · R_armor2world
  //   t_armor2camera = R_camera2gimbalᵀ · (R_gimbal2worldᵀ · t − t_camera2gimbal)
  //   X = R·p + t → x = X.x/X.z, y = X.y/X.z → 5 参 plumb_bob 畸变 → 像素
  // 只支持 0/4/5 参畸变与无 skew 内参，其余退回 reproject_armor（正确性优先）
  const std::size_t nd = distort_coeffs_.total();
  if (
    camera_matrix_.at<double>(0, 1) != 0.0 || camera_matrix_.at<double>(1, 0) != 0.0 ||
    (nd != 0 && nd != 4 && nd != 5))
    return reproject_armor(xyz_in_world, yaw, type, name);

  const double fx = camera_matrix_.at<double>(0, 0);
  const double fy = camera_matrix_.at<double>(1, 1);
  const double cx = camera_matrix_.at<double>(0, 2);
  const double cy = camera_matrix_.at<double>(1, 2);
  double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;
  if (nd >= 4) {
    const double * d = distort_coeffs_.ptr<double>();
    k1 = d[0];
    k2 = d[1];
    p1 = d[2];
    p2 = d[3];
    if (nd == 5) k3 = d[4];
  }

  const Eigen::Vector3d ypr(yaw, pitch_of(name), 0.0);
  const Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * tools::rotation_matrix(ypr);
  const Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * xyz_in_world - t_camera2gimbal_);

  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  std::vector<cv::Point2f> image_points(4);
  for (int i = 0; i < 4; i++) {
    const cv::Point3f & p = object_points[i];
    const Eigen::Vector3d v = R_armor2camera * Eigen::Vector3d(p.x, p.y, p.z) + t_armor2camera;
    const double x = v.x() / v.z();
    const double y = v.y() / v.z();
    const double r2 = x * x + y * y;
    const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    image_points[i] = cv::Point2f((float)(fx * xd + cx), (float)(fy * yd + cy));
  }
  return image_points;
}

double Solver::armor_reprojection_error(const Armor & armor, double yaw) const
{
  // 快路径（见 project_armor_fast；等价性由 solver_test --mode=proj 逐值比对保证）。
  // 原实现每次调 cv::projectPoints（Rodrigues + Mat 分配），140 次 ≈0.93ms，
  // 是单块解算的主要开销（bench）
  const auto image_points = project_armor_fast(armor.xyz_in_world, yaw, armor.type, armor.name);
  double error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

double Solver::pitch_of(ArmorName name) const
{
  return (name == ArmorName::outpost) ? outpost_pitch_ : armor_pitch_;
}

double Solver::yaw_center_of(const Armor & armor) const
{
  const double pnp_yaw = armor.ypr_in_world[0];
  if (yaw_search_center_ == "pnp") return pnp_yaw;

  const double gimbal_yaw = tools::eulers(R_gimbal2world_, 2, 1, 0)[0];
  if (yaw_search_center_ == "gimbal") return gimbal_yaw;

  // bearing：目标方位角（从 PnP 位置算，与 xyz2ypd 的第一个分量同定义）。
  // 无翻转歧义，比 PnP 的 yaw 稳得多——斜视 + 噪声时 PnP 的 yaw 会翻到另一个解
  // （合成扫描：3m/σ=1px/板转 30° 时 PnP yaw 误差均值 26°，方位角则不受影响）
  const double bearing = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  if (yaw_search_center_ == "bearing") return bearing;

  // auto：有 IMU 姿态（调过 set_R_gimbal2world）→ 云台 yaw（同济行为）；
  // 无 IMU → 目标方位角。稳态下装甲板大致朝向相机，板 yaw ≈ 方位角
  const double prior = world_attitude_set_ ? gimbal_yaw : bearing;
  // 兜底：先验窗盖不住 PnP yaw 时以 PnP yaw 为中心，保证真解在搜索窗内
  // （大偏角真值——板转 60° 且偏离光轴 20°；或 PnP 翻转、IMU 姿态失准）。
  // 判据用 68° 而不是 70°：搜索栅格覆盖 center-70° … center+69°，留 1° 余量
  constexpr double GUARD_DEG = 68.0;
  if (std::abs(tools::limit_rad(pnp_yaw - prior)) > GUARD_DEG * CV_PI / 180.0) return pnp_yaw;
  return prior;
}

}  // namespace auto_aim
