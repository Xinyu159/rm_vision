#include "auto_aim/armor.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace auto_aim
{
namespace
{
// 斜视折算的可信门槛。Δh（两灯条像素长之差）是"深度差"信号：Δh/灯条平均长 = W·sinψ/d，
// 距离一远就掉到关键点噪声以下——3m 处小装甲板斜 45° 的 Δh 也只有 ~1px。门槛取
// 5px ≈ 关键点 σ≈1px 时的 3.5σ；再配合 |ψ| 上限，远距离噪声几乎不可能骗开折算
// （3m 小板的两个条件互斥：Δh≥5px 则估出的 ψ>70°，见 --mode=type 扫描）
constexpr double MIN_DH_PX = 5.0;
constexpr double MAX_PSI_DEG = 70.0;  // 超过后 cosψ 太小，折算对 ψ 的误差过于敏感
}  // namespace

ArmorTypeFeatures armor_type_features(const std::vector<cv::Point2f> & p, double fx_px)
{
  ArmorTypeFeatures f;
  if (p.size() != 4) return f;

  const double left = cv::norm(p[0] - p[3]);
  const double right = cv::norm(p[1] - p[2]);
  const double h_mean = (left + right) / 2;
  const double h_max = std::max(left, right);
  const double w = std::max(cv::norm(p[0] - p[1]), cv::norm(p[3] - p[2]));

  f.dh_px = std::abs(left - right);
  f.ratio_max = h_max > 0 ? w / h_max : 0;
  f.ratio_mean = h_mean > 0 ? w / h_mean : 0;
  f.ratio_used = f.ratio_mean;

  if (fx_px <= 0 || h_mean <= 0 || w <= 0) return f;

  // 与型别无关的斜视角：两灯条深度差 = W·sinψ ⇒ Δh/h̄ = W·sinψ/d，
  // 又 W_px = fx·W·cosψ/d，两式相除把 W 约掉：tanψ = (Δh/h̄)·fx/W_px
  f.psi_deg = std::atan(f.dh_px / h_mean * fx_px / w) * 180.0 / CV_PI;

  f.oblique_ok = f.dh_px >= MIN_DH_PX && std::abs(f.psi_deg) <= MAX_PSI_DEG;
  if (f.oblique_ok) f.ratio_used = f.ratio_mean / std::cos(f.psi_deg * CV_PI / 180.0);
  return f;
}

ArmorType infer_armor_type(
  int num_id, const std::vector<cv::Point2f> & points, double big_ratio_threshold, double fx_px)
{
  if (num_id == 8) return ArmorType::big;    // Bb 直接给
  if (num_id == 7) return ArmorType::small;  // Bs 直接给
  return armor_type_features(points, fx_px).ratio_used >= big_ratio_threshold ? ArmorType::big
                                                                             : ArmorType::small;
}

Armor::Armor(
  int color_id, int num_id, float confidence, const cv::Rect & box,
  std::vector<cv::Point2f> armor_keypoints, double big_ratio_threshold, double fx_px,
  bool has_big_armor)
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

  // 型别：场上没有大装甲板时（今年规则）直接按小处理——编号 8(Bb) 例外。误判风险为零，
  // 也不需要关键点几何。有旧大板时（has_big_armor=true）才走几何判定：
  // 旧规则 max(边)/max(边) ≥1.4 在正视时小装甲板也有 2.41（大 4.11），会把小判成大
  // → yaw 假 40°（task #35）。新规则分母用两灯条平均长（near 灯条被放大，用 max 会让
  // 比值系统性偏小），阈值 3.3 取 2.41 与 4.11 的中点；斜视时先用灯条差折算
  type = has_big_armor ? infer_armor_type(num_id, points, big_ratio_threshold, fx_px)
                       : (num_id == 8 ? ArmorType::big : ArmorType::small);
}

}  // namespace auto_aim
