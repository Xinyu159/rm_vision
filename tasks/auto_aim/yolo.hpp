#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

// YOLO 检测（TensorRT 后端，0526 模型）
//
// 对照同济 tasks/auto_aim/yolo.hpp + yolos/yolov5.cpp：
//   - 预处理 / 输出解析逐行对齐（映射表见 docs/0526-model-verification.md §3）
//   - 后端由 OpenVINO IR 换为 TensorRT FP16（engine 由 trtexec 构建，见该文档 §4）
//   - 已含动态 ROI（detect(img, roi)，对齐同济 offset_ 回填模式）；
//     未含 use_traditional 传统法二次矫正（到相应阶段再补）
//
// 已知 engine 绑定（models/0526_fp16.engine）：images fp16 [1,3,640,640] → output fp32 [1,25200,22]

#include <list>
#include <memory>
#include <string>

#include "auto_aim/armor.hpp"

namespace auto_aim
{

struct YoloConfig
{
  std::string engine_path = "models/0526_fp16.engine";  // 相对仓库根（与 yaml 同约定）
  double min_confidence = 0.50;      // objectness 阈值（实测验证值；同济常量 0.7 会漏检）
  double nms_threshold = 0.45;       // 同济用 0.3
  // 场上有无大装甲板：今年规则无大板 → false（所有板按小处理，仅 Bb 例外），型别
  // 误判风险为零；练车遇旧大板时置 true，恢复下面的几何判定
  bool has_big_armor = false;
  // 大/小装甲判定比值阈值（仅 has_big_armor=true 时用）：正对小 2.41 / 大 4.11
  // （max 上下边 ÷ 两灯条平均长），阈值取中点；斜视由灯条差折算后再比
  // （细则见 armor.cpp）。Bs/Bb 由编号直接给出
  double big_ratio_threshold = 3.3;
  double fx = 0;  // 焦距（px，读 yaml 顶层 camera_matrix[0]）；>0 时型别判定启用斜视折算
};

YoloConfig yolo_config_from_yaml(const std::string & config_path);

class YOLO
{
public:
  /// 载入 engine（反序列化 + 分配显存）。失败抛 std::runtime_error，信息里带原因。
  explicit YOLO(const YoloConfig & config);
  ~YOLO();

  /// img 可传单通道 Bayer（相机原始帧，内部走 COLOR_BayerBG2BGR）或三通道 BGR；
  /// 返回装甲板列表（已做置信度过滤 + NMS）。非线程安全（内部复用缓存）。
  std::list<Armor> detect(const cv::Mat & img);

  /// 带 ROI 的检测：只检测 img 的 roi 子区（空/无效 rect == 全图，走上面的等价路径）。
  /// 对齐同济 yolos/{yolov5,yolov8,yolo11}.cpp 的 offset_ 模式：先裁 ROI 再变换到网络
  /// 输入，解码后把关键点/框/中心加回 roi.tl()——返回坐标一律是"整图系"，与全图检测
  /// 语义完全一致（solver/选目标/绘制零改动）。单通道 Bayer 帧会先把 ROI 原点对齐到
  /// 偶坐标（Bayer 2x2 相位），且只转换 ROI 那一块。用途：云台跟随按上一帧目标框扩边
  /// 急裁（见 gimbal_follow）——预处理开销随 ROI 面积缩小，网络输入里目标占比更大。
  std::list<Armor> detect(const cv::Mat & img, const cv::Rect & roi);

  /// 最近一次 detect 的耗时分解（毫秒）
  struct Timings
  {
    double preprocess = 0;
    double inference = 0;
    double postprocess = 0;
    double total() const { return preprocess + inference + postprocess; }
  };
  Timings last_timings() const { return timings_; }
  double last_scale() const { return scale_; }  // 最近一次的 letterbox 缩放比（调试用）

private:
  YoloConfig config_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Timings timings_;
  double scale_ = 1.0;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP
