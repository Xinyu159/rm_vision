#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

// 装甲板数据结构。语义对齐同济 sp_vision_25 tasks/auto_aim/armor.hpp：
// 检测字段由 YOLO 填，位姿字段（solver 阶段）由 Solver::solve 填。
//
// 与同济的差异（依据 docs/0526-model-verification.md §3）：
//   - 编号 8 = 基地大装甲 Bb（同济当 not_armor），此处映射为 base + big
//   - 大/小装甲不能照搬任何编号规则：0526 只有 7/8 (Bs/Bb) 直接给大小；
//     其余编号（含 3/4/5——参考实现 armor_properties 里三/四/五号各有 small
//     与 big 两套类，同号板两种大小都存在）只能从关键点几何反推，
//     细则见 armor_type_features / infer_armor_type

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{
enum Color
{
  red,
  blue,
  extinguish,
  purple
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};

enum ArmorType
{
  big,
  small
};
const std::vector<std::string> ARMOR_TYPES = {"big", "small"};

enum ArmorName
{
  one,
  two,
  three,
  four,
  five,
  sentry,
  outpost,
  base,
  not_armor
};
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};

/// 型别判定的几何特征（由 4 个关键点算；fx <= 0 时不做斜视折算）
struct ArmorTypeFeatures
{
  double ratio_max = 0;    // max(上下边)/max(左右边)——旧规则的比值
  double ratio_mean = 0;   // max(上下边)/两条灯条平均长——与斜视几何自洽的基线
  double dh_px = 0;        // 两条灯条像素长之差（深度差信号）
  double psi_deg = 0;      // 由灯条差估的斜视角（与型别无关）
  bool oblique_ok = false;  // 折算是否可信（Δh 够大、ψ 未到边缘）
  double ratio_used = 0;    // 实际用于判定的比值（折算可信时为 ratio_mean / cosψ）
};

/// 由 4 个关键点算型别特征。fx_px = 焦距（像素），<=0 时只算比值不折算
ArmorTypeFeatures armor_type_features(const std::vector<cv::Point2f> & points, double fx_px);

/// 型别判定：num_id 7/8（Bs/Bb）由编号直接给；其余（含 3/4/5，同号小大都有）
/// 按 big_ratio_threshold 与几何特征推（细则见 armor.cpp）
ArmorType infer_armor_type(
  int num_id, const std::vector<cv::Point2f> & points, double big_ratio_threshold, double fx_px);

struct Armor
{
  Color color;
  ArmorType type;
  ArmorName name;

  int color_id;  // 模型原始颜色类编号（0526 实测：0=蓝 1=红 2=灰 3=紫）
  int num_id;    // 模型原始编号类编号（0=G 1-5=一~五号 6=O 7=Bs 8=Bb）
  float confidence;
  cv::Rect box;                     // 四点外接矩形（NMS 用，非紧致框）
  std::vector<cv::Point2f> points;  // [左上,右上,右下,左下]，已缩放回原图坐标

  cv::Point2f center;       // 四点均值（同济注明：不是对角线交点，不能当实际中心）
  cv::Point2f center_norm;  // 归一化坐标（0~1），由检测层填
  double ratio;              // max(上下边)/max(左右边)，正交时小装甲 ≈2.4、大装甲 ≈4.1
  double rectangular_error;  // 灯条与中点连线夹角和 π/2 之差（对齐同济）

  // ---- 位姿（Solver::solve 填；solved=false 时以下字段未定义）----
  // 坐标系（对齐同济）：x 前 / y 左 / z 上；装甲板自身 x 为板法向的反向
  bool solved = false;
  Eigen::Vector3d xyz_in_gimbal = Eigen::Vector3d::Zero();
  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypr_in_gimbal = Eigen::Vector3d::Zero();  // [yaw, pitch, roll]
  Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();  // [yaw, pitch, distance]
  double yaw_raw = 0.0;  // yaw 精搜索前的 PnP 原始 yaw（同济：平衡步兵分支未初始化，本移植已修）

  /// 由 YOLO 检测结果构造。armor_keypoints 顺序必须是 [左上,右上,右下,左下]。
  /// big_ratio_threshold：nonBs/Bb 编号时，判定比值 ≥ 此值判为大装甲；
  /// fx_px：焦距（像素），>0 时型别判定先做与型别无关的斜视折算；
  /// has_big_armor：场上有无大装甲板。今年规则无大板 → false（所有板按小处理，仅
  /// 编号 8=Bb 例外），误判风险为零；练车/旧场地见大板时置 true 恢复几何判定
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints, double big_ratio_threshold = 3.3,
    double fx_px = 0.0, bool has_big_armor = false);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_HPP
