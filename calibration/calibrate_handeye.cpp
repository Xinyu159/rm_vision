// 定板手眼标定（cv::calibrateHandEye，默认 TSAI）——棋盘格版，移植自同济 sp_vision_25
//
// 输入：capture 采的成对数据 {input-folder}/{n}.jpg + {n}.txt（四元数 wxyz，编号从 1 连续）
// 输出：YAML 到 stdout（R_gimbal2imubody / R_camera2gimbal / t_camera2gimbal + 相机偏角注释）——
//       重定向留档，然后把 R_camera2gimbal / t_camera2gimbal 两行覆盖进 configs/standard3.yaml
//
// 原理：板固定在场地不动，云台带相机转 ≥15 个姿态采图。
//   ① 每张图 solvePnP(IPPE) 得 板→相机 (rvec, tvec，mm)
//   ② 四元数在世界系里"重挂"得 云台→世界：R = R_gimbal2imubody^T · R(q) · R_gimbal2imubody（t=0，云台原地转）
//   ③ calibrateHandEye 解 AX=XB 得 相机→云台
//
// 注意：画面叠加文字只能 ASCII（中文渲染成 "??"）；参数用 = 传值
// 用法：
//   ./calibrate_handeye                        # 默认 assets/img_with_q + configs/calibration.yaml
//   ./calibrate_handeye --display=1            # 逐张开窗看棋盘识别（按任意键继续；需显示器）
//   ./calibrate_handeye --dump=assets/ann      # 把逐张标注图写进目录（无显示器时事后看）
//   ./calibrate_handeye > /tmp/handeye.yaml    # 结果留档
//
// 一张都识别不出：核对 calibration.yaml 的 pattern_cols/pattern_rows
//（11×8 方格 → 内角点 10×7；若板是 11×8 内角点则改 11/8）

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在 opencv2/core/eigen.hpp 上面
#include <cmath>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "tools/fs_tools.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace
{

const cv::String keys =
  "{help h usage ? |      | 打印帮助（参数用 = 传值，如 --dump=assets/ann）}"
  "{config-path c  | configs/calibration.yaml | 配置文件（pattern_*/内参/R_gimbal2imubody）}"
  "{@input-folder  | assets/img_with_q | 输入目录（编号.jpg + 编号.txt，编号从 1 连续）}"
  "{display        | 0    | 1 = 逐张开窗显示识别结果（需显示器，按任意键继续）}"
  "{dump           |      | 目录：把逐张标注图写到该目录（不给就不写）}";

/// OpenCV 解析器把裸 `--flag` 存成字符串 "true"（get<int> 解析 "true" 失败会静默变 0！）——
/// 统一按字符串判：--display / --display=1 / --display=true 都生效，--display=0 关闭
bool flag_on(const cv::CommandLineParser & cli, const std::string & name)
{
  cv::String v = cli.get<cv::String>(name);
  return v == "true" || v == "1" || v == "yes";
}

/// 板面 z=0 的角点坐标（mm）。原点取板上任一角点即可——手眼里用的是板姿态的相对关系
std::vector<cv::Point3f> board_points_3d(const cv::Size & pattern_size, double center_distance_mm)
{
  std::vector<cv::Point3f> points;
  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      points.push_back(
        {(float)(j * center_distance_mm), (float)(i * center_distance_mm), 0});
  return points;
}

bool read_q(const std::string & q_path, Eigen::Quaterniond & q)
{
  std::ifstream q_file(q_path);
  if (!q_file) return false;
  double w, x, y, z;
  if (!(q_file >> w >> x >> y >> z)) return false;
  q = Eigen::Quaterniond(w, x, y, z);
  return true;
}

/// 棋盘识别：SB 变体全分辨率 + EXHAUSTIVE|ACCURACY（亚像素亚级精度、抗噪声，
/// 实测 ~1s/张——离线可接受；经典 findChessboardCorners 实测 ~4.3s 且无板时更不稳，
/// 故偏离同济（同济用圆点靶 + 经典版）统一改用 SB）。
bool detect_board(
  const cv::Mat & bgr, const cv::Size & pattern_size, std::vector<cv::Point2f> & corners)
{
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  corners.clear();
  return cv::findChessboardCornersSB(
    gray, pattern_size, corners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
}

struct Loaded
{
  std::vector<double> R_gimbal2imubody_data;
  std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs;
  int total = 0, used = 0, no_q = 0, no_board = 0;
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
  data.R_gimbal2imubody_data = tools::read<std::vector<double>>(yaml, "R_gimbal2imubody");
  auto camera_matrix_data = tools::read<std::vector<double>>(yaml, "camera_matrix");
  auto distort_coeffs_data = tools::read<std::vector<double>>(yaml, "distort_coeffs");
  if (data.R_gimbal2imubody_data.size() != 9) throw std::runtime_error("R_gimbal2imubody 应为 9 个数");
  if (camera_matrix_data.size() != 9) throw std::runtime_error("camera_matrix 应为 9 个数");
  if (distort_coeffs_data.size() != 5) throw std::runtime_error("distort_coeffs 应为 5 个数");

  cv::Size pattern_size(pattern_cols, pattern_rows);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(data.R_gimbal2imubody_data.data());
  cv::Matx33d camera_matrix(camera_matrix_data.data());
  cv::Mat distort_coeffs(distort_coeffs_data);
  auto points_3d = board_points_3d(pattern_size, center_distance_mm);

  tools::logger()->info(
    "[load] 棋盘内角点 {}x{}、格距 {}mm，输入目录 {}", pattern_size.width, pattern_size.height,
    center_distance_mm, input_folder);

  if (!dump.empty() && !tools::mkdirs(dump))
    throw std::runtime_error("建 dump 目录失败: " + dump);

  for (int i = 1; true; i++) {
    // 编号从 1 连续（capture 从最大编号续采，天然连续）；读不到 {i}.jpg 即结束
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto q_path = fmt::format("{}/{}.txt", input_folder, i);
    auto img = cv::imread(img_path);
    if (img.empty()) break;
    data.total++;

    Eigen::Quaterniond q;
    if (!read_q(q_path, q)) {
      tools::logger()->warn("[load] {} 缺/坏四元数（{}），跳过——--no-q 采的图不能用于手眼", img_path, q_path);
      data.no_q++;
      continue;
    }

    // 云台欧拉角（检查 R_gimbal2imubody 对不对：转云台时 yaw/pitch 应变、roll 应稳）
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * q.toRotationMatrix() * R_gimbal2imubody;
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;  // degree

    auto drawing = img.clone();
    tools::draw_text(drawing, fmt::format("yaw   {:.2f}", ypr[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("pitch {:.2f}", ypr[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("roll  {:.2f}", ypr[2]), {40, 120}, {0, 0, 255});

    std::vector<cv::Point2f> corners;
    bool ok = detect_board(img, pattern_size, corners);
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

    // 板→相机（mm）；云台在世界系里原地转 → t_gimbal2world = 0
    cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat R_gimbal2world_cv;
    cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
    cv::Mat rvec, tvec;
    cv::solvePnP(
      points_3d, corners, camera_matrix, distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);

    data.R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
    data.t_gimbal2world_list.emplace_back(t_gimbal2world);
    data.rvecs.emplace_back(rvec);
    data.tvecs.emplace_back(tvec);
    data.used++;
  }
  return data;
}

void print_yaml(
  const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
  const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr)
{
  YAML::Emitter result;
  std::vector<double> R_camera2gimbal_data(
    R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
  std::vector<double> t_camera2gimbal_data(
    t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

  result << YAML::BeginMap;
  result << YAML::Key << "R_gimbal2imubody";
  result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
  result << YAML::Newline;
  result << YAML::Newline;
  result << YAML::Comment(fmt::format(
    "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));
  result << YAML::Key << "R_camera2gimbal";
  result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
  result << YAML::Key << "t_camera2gimbal";
  result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
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
    tools::logger()->error("[calibrate_handeye] 加载失败: {}", e.what());
    return 1;
  }

  if (data.used == 0) {
    tools::logger()->error(
      "[calibrate_handeye] 一张都没识别出棋盘——核对 calibration.yaml 的 pattern_cols/pattern_rows"
      "（11x8 方格 → 内角点 10x7）与板面清晰度；上次采图目录对不对？");
    return 1;
  }
  if (data.used < 3) {
    tools::logger()->error("[calibrate_handeye] 有效数据只有 {} 组（建议 ≥15），先补采", data.used);
    return 1;
  }
  tools::logger()->info(
    "[calibrate_handeye] 共 {} 张：识别成功 {}，缺四元数 {}，未识别 {}", data.total, data.used,
    data.no_q, data.no_board);

  // 定板手眼：AX=XB（板在云台自转的世界系中不动）
  cv::Mat R_camera2gimbal, t_camera2gimbal;
  cv::calibrateHandEye(
    data.R_gimbal2world_list, data.t_gimbal2world_list, data.rvecs, data.tvecs, R_camera2gimbal,
    t_camera2gimbal);
  t_camera2gimbal /= 1e3;  // mm → m

  // 相机同"理想正装"的偏角（就是"相机怎么装的"：yaw/pitch/roll，度）
  Eigen::Matrix3d R_camera2gimbal_eigen;
  cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
  Eigen::Matrix3d R_gimbal2ideal;  // 相机正装理论值（同济用嵌套初始化列表；Eigen 3.3 不支持，改逗号初始化）
  R_gimbal2ideal << 0, -1, 0, 0, 0, -1, 1, 0, 0;
  Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
  Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // degree

  tools::logger()->info(
    "[calibrate_handeye] 相机同理想安装偏角: yaw {:.2f}  pitch {:.2f}  roll {:.2f} (deg)；"
    "相机→云台距离 {:.4f} m",
    ypr[0], ypr[1], ypr[2], cv::norm(t_camera2gimbal));

  print_yaml(data.R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);
  tools::logger()->info(
    "[calibrate_handeye] 把上面 R_camera2gimbal / t_camera2gimbal 两行覆盖进 configs/standard3.yaml");
  return 0;
}
