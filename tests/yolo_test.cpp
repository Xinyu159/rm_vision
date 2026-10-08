// YOLO 检测离线测试：对单张图片跑 0526 检测，打印每个装甲板的解码结果与耗时
// （不开相机、不出画面——用来和 tools/verify_0526.py 的 python 参考结果对拍）
//
// 用法：
//   ./build/yolo_test docs/img/0526_demo_field.jpg
//   ./build/yolo_test --conf=0.6 --nms=0.4 path/to/frame.jpg

#include <algorithm>
#include <cstdio>
#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/logger.hpp"

int main(int argc, char ** argv)
{
  const cv::String keys =
    "{help h usage ? |      | 打印帮助 }"
    "{@image        |      | 输入图片路径（必填）}"
    "{config        | configs/standard3.yaml | 配置文件（读 yolo: 块）}"
    "{conf c        | 0    | objectness 阈值（>0 覆盖 yaml）}"
    "{nms n         | 0    | NMS 阈值（>0 覆盖 yaml）}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || cli.get<cv::String>("@image").empty()) {
    cli.printMessage();
    return cli.has("help") ? 0 : 1;
  }

  const std::string image_path = cli.get<cv::String>("@image");
  const std::string config_path = cli.get<cv::String>("config");

  auto config = auto_aim::yolo_config_from_yaml(config_path);
  double conf = cli.get<double>("conf");
  if (conf > 0) config.min_confidence = conf;
  double nms = cli.get<double>("nms");
  if (nms > 0) config.nms_threshold = nms;

  auto_aim::YOLO yolo(config);

  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    tools::logger()->error("[yolo_test] 读不到图片: {}", image_path);
    return 1;
  }
  tools::logger()->info("[yolo_test] 输入 {}  {}x{}", image_path, img.cols, img.rows);

  // 首次 detect 含 CUDA 上下文 / OpenCV 线程池的懒初始化（实测 ~100ms 起），
  // 预热 3 次后跑 30 次取分位数——单次计时全是冷启动/首帧噪声
  for (int i = 0; i < 3; i++) yolo.detect(img);

  constexpr int N = 30;
  std::vector<double> pre, infer, post;
  std::list<auto_aim::Armor> armors;
  for (int i = 0; i < N; i++) {
    armors = yolo.detect(img);
    const auto t = yolo.last_timings();
    pre.push_back(t.preprocess);
    infer.push_back(t.inference);
    post.push_back(t.postprocess);
  }
  auto pct = [](std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[(std::size_t)(p * (v.size() - 1))];
  };

  std::printf(
    "\n检出 %zu 个装甲板  缩放 %.4f  （预热 3 次后 %d 次分位数）\n"
    "  预处理 p50=%.2f p99=%.2f | 推理 p50=%.2f p99=%.2f | 解码 p50=%.2f p99=%.2f ms\n\n",
    armors.size(), yolo.last_scale(), N, pct(pre, 0.5), pct(pre, 0.99), pct(infer, 0.5),
    pct(infer, 0.99), pct(post, 0.5), pct(post, 0.99));
  for (const auto & a : armors) {
    std::printf(
      "%-10s %-8s %-5s conf=%.3f  color_id=%d num_id=%d  ratio=%.2f rect_err=%.3f\n"
      "    box=(%d,%d %dx%d)  center=(%.1f,%.1f)  norm=(%.3f,%.3f)\n"
      "    pts TL=(%.1f,%.1f) TR=(%.1f,%.1f) BR=(%.1f,%.1f) BL=(%.1f,%.1f)\n",
      auto_aim::COLORS[a.color].c_str(), auto_aim::ARMOR_NAMES[a.name].c_str(),
      auto_aim::ARMOR_TYPES[a.type].c_str(), a.confidence, a.color_id, a.num_id, a.ratio,
      a.rectangular_error, a.box.x, a.box.y, a.box.width, a.box.height, a.center.x, a.center.y,
      a.center_norm.x, a.center_norm.y, a.points[0].x, a.points[0].y, a.points[1].x, a.points[1].y,
      a.points[2].x, a.points[2].y, a.points[3].x, a.points[3].y);
  }
  return armors.empty() ? 2 : 0;  // 无检出退出码 2（脚本对拍时便于区分）
}
