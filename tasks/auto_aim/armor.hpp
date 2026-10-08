#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

// 装甲板数据结构。语义对齐同济 sp_vision_25 tasks/auto_aim/armor.hpp，
// 只保留检测阶段用得到的字段（PnP/EKF 的位姿字段等到解算/跟踪阶段再补）。
//
// 与同济的差异（依据 docs/0526-model-verification.md §3）：
//   - 编号 8 = 基地大装甲 Bb（同济当 not_armor），此处映射为 base + big
//   - 大/小装甲不照搬同济的 num_id==1→big（0526 编号体系不同），
//     改由 Bs/Bb 编号或关键点几何（ratio）判定

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
  double ratio;              // 长边/宽边（正对时小装甲 ≈1.1，大装甲 ≈1.8）
  double rectangular_error;  // 灯条与中点连线夹角和 π/2 之差（对齐同济）

  /// 由 YOLO 检测结果构造。armor_keypoints 顺序必须是 [左上,右上,右下,左下]。
  /// big_ratio_threshold：非 Bs/Bb 编号时，ratio ≥ 此值判为大装甲
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints, double big_ratio_threshold = 1.4);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_HPP
