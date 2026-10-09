#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

// PnP 位姿解算（移植 sp_vision_25 tasks/auto_aim/solver.{hpp,cpp}）
//
// 链路：solvePnP(SOLVEPNP_IPPE) 得相机系 R,t
//       → 相机系 → gimbal 系 → world 系（级联外参）
//       → eulers(ZYX) 得 ypr、xyz2ypd 得球坐标
//       → yaw 精搜索（pitch 固定假设、roll=0，中心 ±70° 内 1° 步长重投影穷举）覆盖 ypr[0]
// 坐标系约定（对齐同济）：x 前 / y 左 / z 上；装甲板本体系 x 为板法向的反向（板面在 x=0）
//
// 与同济的差异（决策与依据见 docs/conclusions-and-open-items.md）：
//   1. 配置走本仓库 *Config + *_config_from_yaml 模式（yaml 键名与同济一致）
//   2. 外参缺省 = 相机理想正装旋转，**不是单位阵**——单位阵等于把相机系当世界系
//      （光轴=z 轴），ypr/ypd 参数化在目标正前方时退化（yaw≡0、pitch≡90°），静默出错
//   3. yaw 搜索中心可配：auto（默认）= 有 IMU → 云台 yaw（同济行为）；无 IMU → 目标方位角
//      （比 PnP 的 yaw 稳：斜视+噪声时 PnP 会翻解）；另可强制 pnp / gimbal / bearing。
//      auto 带兜底：先验窗盖不住 PnP yaw 时退回 PnP yaw（正常 FOV 内观测不触发）
//   4. solve() 返回 bool，失败时不写 armor 位姿字段；平衡步兵分支补 yaw_raw 初始化
//      （同济那处是未初始化内存）
//   5. 不移植死代码：SJTU_cost / oupost_reprojection_error（调用处已注释）、world2pixel
//      （到用得上的阶段再加）
//   6. yaw 精搜索内循环用等价手写投影（140 次 cv::projectPoints ≈0.93ms → ≈0.03ms）；
//      等价性由 solver_test --mode=proj 逐值比对（点差 ≤0.01px、搜索 argmin 一致）

#include <Eigen/Geometry>          // 必须在 opencv2/core/eigen.hpp 上面
#include <opencv2/core/eigen.hpp>  // cv::eigen2cv / cv2eigen
#include <string>
#include <vector>

#include "auto_aim/armor.hpp"

namespace auto_aim
{

/// 相机理想正装（光轴水平朝前、无滚转）的 camera→gimbal 旋转，行主序。
/// 与同济 mvs 配置的理论外参相同；不要用单位阵（见文件头注释 2）
inline const std::vector<double> R_CAMERA2GIMBAL_NOMINAL = {0, 0, 1, -1, 0, 0, 0, -1, 0};

struct SolverConfig
{
  std::vector<double> camera_matrix;   // 9，行主序（yaml 顶层，必填）
  std::vector<double> distort_coeffs;  // 5（yaml 顶层，必填）
  std::vector<double> R_gimbal2imubody = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 9，行主序
  std::vector<double> R_camera2gimbal = R_CAMERA2GIMBAL_NOMINAL;      // 9，行主序
  std::vector<double> t_camera2gimbal = {0, 0, 0};                   // 3，单位 m
  std::string yaw_search_center = "auto";  // auto / pnp / gimbal / bearing（见文件头注释 3）
  double armor_pitch_deg = 15.0;           // yaw 精搜索的 pitch 假设（同济值）
  double outpost_pitch_deg = -15.0;        // 前哨站 pitch（同济值）
};

/// 读 yaml：camera_matrix / distort_coeffs 必填，其余键缺省（缺省值见结构体）
SolverConfig solver_config_from_yaml(const std::string & config_path);

class Solver
{
public:
  explicit Solver(const SolverConfig & config);

  /// 解一块装甲板：成功填 armor 的位姿字段并置 solved=true，返回 true；
  /// 失败（点数不对 / solvePnP 失败 / 位置非有限 / 板在相机后方）返回 false 且不改字段
  bool solve(Armor & armor) const;

  /// 用（世界系位置, yaw）重投影 4 点（pitch 用配置假设、roll=0）。测试 / 画图用。
  /// 实现走 cv::projectPoints，是快路径的参考实现与 --mode=proj 的比对锚点
  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  /// 投影快路径：与 reproject_armor 逐项等价的手写投影，供 yaw 精搜索内循环用
  /// （免掉每次 cv::projectPoints 的 Rodrigues + 缓冲分配）。只支持 0/4/5 参
  /// plumb_bob 畸变与零 skew 的内参，其余情况自动退回 reproject_armor
  std::vector<cv::Point2f> project_armor_fast(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  Eigen::Matrix3d R_gimbal2world() const;

  /// IMU 姿态接入后每帧调用（对齐同济）；调用后 yaw_search_center=auto 切到云台 yaw 为中心
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  struct Timings
  {
    double pnp = 0;         // ms：solvePnP + 坐标级联 + eulers/ypd
    double yaw_search = 0;  // ms：yaw 精搜索（140 次重投影）
    double total() const { return pnp + yaw_search; }
  };
  /// 最近一次 solve() 的耗时分解（一帧多块装甲板时是最后一块的）
  Timings last_timings() const { return timings_; }

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();
  bool world_attitude_set_ = false;  // 是否调过 set_R_gimbal2world（即是否已有 IMU 姿态）
  std::string yaw_search_center_;
  double armor_pitch_;    // rad
  double outpost_pitch_;  // rad
  mutable Timings timings_;

  void optimize_yaw(Armor & armor) const;
  double armor_reprojection_error(const Armor & armor, double yaw) const;
  double pitch_of(ArmorName name) const;
  double yaw_center_of(const Armor & armor) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP
