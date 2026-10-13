// 相机内参标定（棋盘格版，移植自同济 calibrate_camera；识别/无头化对齐本仓库）
//
// 输入：capture 采的图 {input-folder}/{n}.jpg（编号从 1 连续；与手眼同一批数据可直接复用）
// 输出：YAML 到 stdout（camera_matrix + distort_coeffs + 重投影误差注释）——
//       重定向留档，然后把两行覆盖进 configs/calibration.yaml（与 standard3.yaml 相机块同步）
//
// 原理：板固定/手举皆可，只要图里板姿态足够多样 → calibrateCamera 解内参 + 畸变。
//   与手眼的区别：内参不看四元数（只用图），视角变化在图里体现即可。
//
// 注意：画面叠加文字只能 ASCII（中文渲染成 "??"）；参数用 = 传值
// 用法：
//   ./calibrate_camera                        # 默认 assets/img_with_q + configs/calibration.yaml
//   ./calibrate_camera --display=1            # 逐张开窗看棋盘识别（需显示器，按任意键继续）
//   ./calibrate_camera --dump=assets/ann      # 把逐张标注图写进目录（无显示器时事后看）
//   ./calibrate_camera > /tmp/intr.yaml       # 结果留档
//
// 一张都识别不出：核对 calibration.yaml 的 pattern_cols/pattern_rows
//（本板实测 = 11×8 内角点，即 12×9 方格；见 calibration.yaml 注释）

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "tools/fs_tools.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace
{

const cv::String keys =
  "{help h usage ? |      | 打印帮助（参数用 = 传值，如 --dump=assets/ann）}"
  "{config-path c  | configs/calibration.yaml | 配置文件（pattern_*）}"
  "{@input-folder  | assets/img_with_q | 输入目录（编号.jpg，编号从 1 连续）}"
  "{display        | 0    | 1 = 逐张开窗显示识别结果（需显示器，按任意键继续）}"
  "{dump           |      | 目录：把逐张标注图写到该目录（不给就不写）}";

/// OpenCV 解析器把裸 `--flag` 存成字符串 "true"（get<int> 解析 "true" 失败会静默变 0！）——
/// 统一按字符串判：--display / --display=1 / --display=true 都生效，--display=0 关闭
bool flag_on(const cv::CommandLineParser & cli, const std::string & name)
{
  cv::String v = cli.get<cv::String>(name);
  return v == "true" || v == "1" || v == "yes";
}

/// 板面 z=0 的角点坐标（mm）。原点取板上任一角点即可——内参只看板面几何
std::vector<cv::Point3f> board_points_3d(const cv::Size & pattern_size, double center_distance_mm)
{
  std::vector<cv::Point3f> points;
  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      points.push_back(
        {(float)(j * center_distance_mm), (float)(i * center_distance_mm), 0});
  return points;
}

/// 棋盘识别链与手眼程序一致（见 calibrate_robotworld_handeye.cpp 注释）
bool detect_board(
  const cv::Mat & bgr, const cv::Size & pattern_size, std::vector<cv::Point2f> & corners)
{
  cv::Mat gray, enhanced;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::createCLAHE(3.0, cv::Size(8, 8))->apply(gray, enhanced);

  corners.clear();
  if (cv::findChessboardCornersSB(
        gray, pattern_size, corners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY))
    return true;
  if (cv::findChessboardCornersSB(
        enhanced, pattern_size, corners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY))
    return true;
  // live 端同款（全+CLAHE+plain）：实测个别帧 EXH|ACC 全挂而这步过（如 47.jpg）
  if (cv::findChessboardCornersSB(enhanced, pattern_size, corners)) return true;

  cv::Mat half;
  cv::resize(enhanced, half, {}, 0.5, 0.5, cv::INTER_AREA);
  bool ok = cv::findChessboardCornersSB(half, pattern_size, corners) ||
            cv::findChessboardCornersSB(
              half, pattern_size, corners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
  if (ok)
    for (auto & c : corners) c *= 2.f;
  return ok;
}

struct Loaded
{
  std::vector<std::vector<cv::Point3f>> obj_points;
  std::vector<std::vector<cv::Point2f>> img_points;
  std::vector<int> img_index;  // 每张有效图的原编号（报"最差第几张"用）
  cv::Size img_size;
  int total = 0, used = 0, no_board = 0;
};

Loaded load(
  const std::string & input_folder, const std::string & config_path, bool display,
  const std::string & dump)
{
  Loaded data;
  auto yaml = tools::load(config_path);
  auto pattern_cols = tools::read<int>(yaml, "pattern_cols");
  auto pattern_rows = tools::read<int>(yaml, "pattern_rows");
  auto center_distance_mm = tools::read<double>(yaml, "center_distance_mm");

  cv::Size pattern_size(pattern_cols, pattern_rows);
  auto points_3d = board_points_3d(pattern_size, center_distance_mm);

  tools::logger()->info(
    "[load] 棋盘内角点 {}x{}、格距 {}mm，输入目录 {}", pattern_size.width, pattern_size.height,
    center_distance_mm, input_folder);

  if (!dump.empty() && !tools::mkdirs(dump))
    throw std::runtime_error("建 dump 目录失败: " + dump);

  for (int i = 1; true; i++) {
    // 编号从 1 连续（capture 从最大编号续采，天然连续）；读不到 {i}.jpg 即结束
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto img = cv::imread(img_path);
    if (img.empty()) break;
    data.total++;

    if (data.img_size.width == 0)
      data.img_size = img.size();
    else if (img.size() != data.img_size) {
      tools::logger()->warn("[load] {} 尺寸与其他图不一致，跳过（内参要求同尺寸）", img_path);
      continue;
    }

    std::vector<cv::Point2f> corners;
    bool ok = detect_board(img, pattern_size, corners);

    auto drawing = img.clone();
    cv::drawChessboardCorners(drawing, pattern_size, corners, ok);
    if (!dump.empty()) cv::imwrite(fmt::format("{}/{}.jpg", dump, i), drawing);
    if (display) {
      cv::imshow("Press any key to continue", drawing);
      cv::waitKey(0);
    }

    fmt::print("[{}] {}\n", ok ? "success" : "failure", img_path);
    if (!ok) {
      data.no_board++;
      continue;
    }

    data.img_points.emplace_back(corners);
    data.obj_points.emplace_back(points_3d);
    data.img_index.emplace_back(i);
    data.used++;
  }
  return data;
}

void print_yaml(const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs, double err_mean,
  double rms)
{
  YAML::Emitter result;
  std::vector<double> camera_matrix_data(
    camera_matrix.begin<double>(), camera_matrix.end<double>());
  std::vector<double> distort_coeffs_data(
    distort_coeffs.begin<double>(), distort_coeffs.end<double>());

  result << YAML::BeginMap;
  result << YAML::Comment(fmt::format("重投影误差: mean {:.3f}px / rms {:.3f}px", err_mean, rms));
  result << YAML::Key << "camera_matrix";
  result << YAML::Value << YAML::Flow << camera_matrix_data;
  result << YAML::Key << "distort_coeffs";
  result << YAML::Value << YAML::Flow << distort_coeffs_data;
  result << YAML::Newline;
  result << YAML::EndMap;

  fmt::print("\n{}\n", result.c_str());
}

}  // namespace

int main(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<cv::String>("config-path");
  const std::string input_folder = cli.get<cv::String>("@input-folder");
  const bool display = flag_on(cli, "display");
  const std::string dump = cli.get<cv::String>("dump");

  Loaded data;
  try {
    data = load(input_folder, config_path, display, dump);
  } catch (const std::exception & e) {
    tools::logger()->error("[calibrate_camera] 加载失败: {}", e.what());
    return 1;
  }

  if (data.used == 0) {
    if (data.total == 0)
      tools::logger()->error(
        "[calibrate_camera] 输入目录没有 1.jpg 起的数据——编号必须从 1 连续"
        "（capture 从最大编号续采；中间删过图请重新连号）");
    else
      tools::logger()->error(
        "[calibrate_camera] 一张都没识别出棋盘——核对 calibration.yaml 的 pattern_cols/pattern_rows"
        "与板面清晰度（暗场地先开灯/靠近板）");
    return 1;
  }
  if (data.used < 4) {
    tools::logger()->error("[calibrate_camera] 有效数据只有 {} 组（内参建议 ≥15），先补采", data.used);
    return 1;
  }
  if (data.used < 15)
    tools::logger()->warn("[calibrate_camera] 有效数据只有 {} 组，内参建议 ≥15 更稳", data.used);
  tools::logger()->info(
    "[calibrate_camera] 共 {} 张：识别成功 {}，未识别 {}", data.total, data.used, data.no_board);

  cv::Mat camera_matrix, distort_coeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  auto criteria = cv::TermCriteria(
    cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100,
    DBL_EPSILON);  // 默认迭代次数(30)有时会导致结果发散，故设为100
  // flags=0：k1 k2 p1 p2 k3 全放开（本镜头实测 k3≈-0.12 非小量；不学同济 FIX_K3）
  auto rms = cv::calibrateCamera(
    data.obj_points, data.img_points, data.img_size, camera_matrix, distort_coeffs, rvecs, tvecs,
    0, criteria);

  // 逐张重投影误差 → 平均 + 最差（最差那张多半有运动模糊/角度过斜，可删了重跑）
  double err_sum = 0, err_max = 0;
  size_t total_points = 0;
  int worst_index = 0;
  for (size_t i = 0; i < data.obj_points.size(); i++) {
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(
      data.obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected);

    double err = 0;
    for (size_t j = 0; j < reprojected.size(); j++)
      err += cv::norm(data.img_points[i][j] - reprojected[j]);
    err /= reprojected.size();

    err_sum += err;
    total_points += reprojected.size();
    if (err > err_max) {
      err_max = err;
      worst_index = data.img_index[i];
    }
  }
  auto err_mean = err_sum / data.obj_points.size();

  tools::logger()->info(
    "[calibrate_camera] 重投影误差 mean {:.3f}px / rms {:.3f}px；最差第 {} 张 {:.3f}px", err_mean,
    rms, worst_index, err_max);

  // 和现配置对比（决定要不要更新）
  try {
    auto yaml = tools::load(config_path);
    auto old = tools::read<std::vector<double>>(yaml, "camera_matrix");
    if (old.size() == 9)
      tools::logger()->info(
        "[calibrate_camera] 现配置 fx {:.1f} fy {:.1f} cx {:.1f} cy {:.1f} → 新解 fx {:.1f} fy "
        "{:.1f} cx {:.1f} cy {:.1f}",
        old[0], old[4], old[2], old[5], camera_matrix.at<double>(0, 0),
        camera_matrix.at<double>(1, 1), camera_matrix.at<double>(0, 2),
        camera_matrix.at<double>(1, 2));
  } catch (const std::exception &) {
    // 对比失败不影响标定结果
  }

  print_yaml(camera_matrix, distort_coeffs, err_mean, rms);
  tools::logger()->info(
    "[calibrate_camera] 把上面 camera_matrix / distort_coeffs 两行覆盖进 configs/calibration.yaml"
    "（与 standard3.yaml 相机块同步）");
  return 0;
}
