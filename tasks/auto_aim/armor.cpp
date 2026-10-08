#include "auto_aim/armor.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace auto_aim
{

Armor::Armor(
  int color_id, int num_id, float confidence, const cv::Rect & box,
  std::vector<cv::Point2f> armor_keypoints, double big_ratio_threshold)
: color_id(color_id),
  num_id(num_id),
  confidence(confidence),
  box(box),
  points(std::move(armor_keypoints))
{
  // 以下几何量与同济 armor.cpp 的 (color_id, num_id) 构造函数逐行一致
  center = (points[0] + points[1] + points[2] + points[3]) / 4;
  auto left_width = cv::norm(points[0] - points[3]);
  auto right_width = cv::norm(points[1] - points[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(points[0] - points[1]);
  auto bottom_length = cv::norm(points[3] - points[2]);
  auto max_length = std::max(top_length, bottom_length);
  auto left_center = (points[0] + points[3]) / 2;
  auto right_center = (points[1] + points[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(
    std::atan2((points[3] - points[0]).y, (points[3] - points[0]).x) - roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2((points[2] - points[1]).y, (points[2] - points[1]).x) - roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  ratio = max_width > 0 ? max_length / max_width : 0;  // 退化检测的零保护

  // 0526 颜色类实测：0=蓝 1=红 2=灰(熄灭) 3=紫（同济此处第 3 类落在 extinguish）
  color = color_id == 0  ? Color::blue
          : color_id == 1 ? Color::red
          : color_id == 3 ? Color::purple
                          : Color::extinguish;

  // 0526 编号类：0=G 1-5=一~五号 6=O 7=Bs(基地小) 8=Bb(基地大)
  name = num_id == 0 ? ArmorName::sentry
         : num_id < 6 ? ArmorName(num_id - 1)
         : num_id == 6 ? ArmorName::outpost
                       : ArmorName::base;

  // 型别：Bs/Bb 由编号直接给出；其余按关键点几何
  type = num_id == 8 ? ArmorType::big
         : num_id == 7 ? ArmorType::small
         : (ratio >= big_ratio_threshold ? ArmorType::big : ArmorType::small);
}

}  // namespace auto_aim
