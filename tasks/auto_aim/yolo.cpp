#include "auto_aim/yolo.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <stdexcept>
#include <vector>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace auto_aim
{
namespace
{
/// TensorRT 日志转接（只放行 warning 及以上，避免刷屏）
class TrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kWARNING) tools::logger()->warn("[yolo][TRT] {}", msg);
  }
};

std::size_t dtype_bytes(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    default: return 0;
  }
}

const char * dtype_name(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return "fp32";
    case nvinfer1::DataType::kHALF: return "fp16";
    case nvinfer1::DataType::kINT8: return "int8";
    case nvinfer1::DataType::kINT32: return "int32";
    default: return "?";
  }
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

int argmax(const float * p, int n)
{
  int best = 0;
  for (int i = 1; i < n; i++)
    if (p[i] > p[best]) best = i;
  return best;
}

double delta_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string dims_str(const nvinfer1::Dims & d)
{
  std::string s = "[";
  for (int i = 0; i < d.nbDims; i++) {
    s += std::to_string(d.d[i]);
    if (i + 1 < d.nbDims) s += ",";
  }
  return s + "]";
}

/// 检查静态形状：不接受 -1（动态形状需要优化档，本实现不支持）
void require_static(const nvinfer1::Dims & d, const char * what)
{
  for (int i = 0; i < d.nbDims; i++)
    if (d.d[i] < 0)
      throw std::runtime_error(
        std::string(what) + " 是动态形状 " + dims_str(d) + "，本实现只支持静态形状（trtexec 按固定输入构建）");
}
}  // namespace

struct YOLO::Impl
{
  TrtLogger trt_logger;
  nvinfer1::IRuntime * runtime = nullptr;
  nvinfer1::ICudaEngine * engine = nullptr;
  nvinfer1::IExecutionContext * context = nullptr;
  cudaStream_t stream = nullptr;

  int in_index = -1, out_index = -1;
  nvinfer1::DataType in_dtype = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType out_dtype = nvinfer1::DataType::kFLOAT;
  int input_w = 0, input_h = 0;
  int out_rows = 0, out_cols = 0;
  std::size_t in_bytes = 0, out_bytes = 0;

  void * dev_in = nullptr;
  void * dev_out = nullptr;
  std::vector<void *> bindings;
  std::vector<std::uint8_t> host_out;

  // CPU 侧缓存（避免每帧重新分配）
  cv::Mat bayer_bgr, resized, canvas, blob32, blob16, out_f32;

  ~Impl()
  {
    if (dev_in) cudaFree(dev_in);
    if (dev_out) cudaFree(dev_out);
    if (stream) cudaStreamDestroy(stream);
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
  }
};

YOLO::YOLO(const YoloConfig & config) : config_(config), impl_(new Impl)
{
  std::ifstream f(config_.engine_path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("打不开 engine 文件: " + config_.engine_path);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<char> buf((std::size_t)size);
  if (!f.read(buf.data(), size))
    throw std::runtime_error("读 engine 文件失败: " + config_.engine_path);

  impl_->runtime = nvinfer1::createInferRuntime(impl_->trt_logger);
  if (!impl_->runtime) throw std::runtime_error("createInferRuntime 失败");
  impl_->engine = impl_->runtime->deserializeCudaEngine(buf.data(), buf.size());
  if (!impl_->engine)
    throw std::runtime_error(
      "engine 反序列化失败（TRT 版本或设备不匹配？）: " + config_.engine_path);
  impl_->context = impl_->engine->createExecutionContext();
  if (!impl_->context) throw std::runtime_error("createExecutionContext 失败");
  if (cudaStreamCreate(&impl_->stream) != cudaSuccess) throw std::runtime_error("建 CUDA 流失败");

  const int nb = impl_->engine->getNbBindings();
  if (nb != 2) throw std::runtime_error("期望 2 个绑定（1 输入 1 输出），实得 " + std::to_string(nb));
  for (int i = 0; i < nb; i++) {
    tools::logger()->info(
      "[yolo] 绑定 {}: {} {} {} {}", i, impl_->engine->getBindingName(i),
      impl_->engine->bindingIsInput(i) ? "输入" : "输出", dtype_name(impl_->engine->getBindingDataType(i)),
      dims_str(impl_->engine->getBindingDimensions(i)));
    if (impl_->engine->bindingIsInput(i))
      impl_->in_index = i;
    else
      impl_->out_index = i;
  }

  {
    const auto d = impl_->engine->getBindingDimensions(impl_->in_index);
    require_static(d, "输入");
    if (d.nbDims != 4 || d.d[0] != 1 || d.d[1] != 3)
      throw std::runtime_error("输入形状应为 [1,3,H,W]，实得 " + dims_str(d));
    impl_->in_dtype = impl_->engine->getBindingDataType(impl_->in_index);
    impl_->input_h = d.d[2];
    impl_->input_w = d.d[3];
    impl_->in_bytes = (std::size_t)d.d[1] * d.d[2] * d.d[3] * dtype_bytes(impl_->in_dtype);
  }
  {
    const auto d = impl_->engine->getBindingDimensions(impl_->out_index);
    require_static(d, "输出");
    if (d.nbDims != 3 || d.d[0] != 1)
      throw std::runtime_error("输出形状应为 [1,25200,22]，实得 " + dims_str(d));
    impl_->out_dtype = impl_->engine->getBindingDataType(impl_->out_index);
    impl_->out_rows = d.d[1];
    impl_->out_cols = d.d[2];
    if (impl_->out_cols != 22)
      throw std::runtime_error("输出列数不是 22（实得 " + std::to_string(impl_->out_cols) + "），模型不匹配");
    impl_->out_bytes = (std::size_t)impl_->out_rows * impl_->out_cols * dtype_bytes(impl_->out_dtype);
  }

  if (cudaMalloc(&impl_->dev_in, impl_->in_bytes) != cudaSuccess)
    throw std::runtime_error("cudaMalloc 输入缓冲失败");
  if (cudaMalloc(&impl_->dev_out, impl_->out_bytes) != cudaSuccess)
    throw std::runtime_error("cudaMalloc 输出缓冲失败");
  impl_->host_out.resize(impl_->out_bytes);
  impl_->bindings.assign(nb, nullptr);
  impl_->bindings[impl_->in_index] = impl_->dev_in;
  impl_->bindings[impl_->out_index] = impl_->dev_out;

  tools::logger()->info(
    "[yolo] engine 载入完成: {}  输入 {}x{} {}  输出 {}x{} {}  conf≥{:.2f} nms {:.2f} "
    "big_ratio {:.2f}",
    config_.engine_path, impl_->input_w, impl_->input_h, dtype_name(impl_->in_dtype),
    impl_->out_rows, impl_->out_cols, dtype_name(impl_->out_dtype), config_.min_confidence,
    config_.nms_threshold, config_.big_ratio_threshold);
}

YOLO::~YOLO() = default;

std::list<Armor> YOLO::detect(const cv::Mat & img)
{
  using clock = std::chrono::steady_clock;
  std::list<Armor> armors;
  if (img.empty()) return armors;

  // ---- 预处理（对齐 tools/verify_0526.py：等比缩放到装得下、左上对齐、黑边补齐）----
  auto t0 = clock::now();
  cv::Mat src;
  if (img.channels() == 1) {
    cv::cvtColor(img, impl_->bayer_bgr, cv::COLOR_BayerBG2BGR);
    src = impl_->bayer_bgr;
  } else if (img.channels() == 3) {
    src = img;
  } else {
    throw std::runtime_error("detect: 只支持单通道 Bayer 或三通道 BGR");
  }

  scale_ = std::min((double)impl_->input_h / src.rows, (double)impl_->input_w / src.cols);
  const int nh = (int)(src.rows * scale_);
  const int nw = (int)(src.cols * scale_);

  impl_->canvas.create(impl_->input_h, impl_->input_w, CV_8UC3);
  impl_->canvas.setTo(cv::Scalar(0, 0, 0));
  cv::resize(src, impl_->resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
  impl_->resized.copyTo(impl_->canvas(cv::Rect(0, 0, nw, nh)));

  // BGR→RGB、/255、NCHW 一次完成
  cv::dnn::blobFromImage(
    impl_->canvas, impl_->blob32, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);

  const void * in_host = nullptr;
  if (impl_->in_dtype == nvinfer1::DataType::kHALF) {
    impl_->blob32.convertTo(impl_->blob16, CV_16F);  // 本 engine 输入即 fp16
    in_host = impl_->blob16.ptr();
  } else if (impl_->in_dtype == nvinfer1::DataType::kFLOAT) {
    in_host = impl_->blob32.ptr();
  } else {
    throw std::runtime_error("不支持的输入类型（只支持 fp32/fp16）");
  }
  if (cudaMemcpy(impl_->dev_in, in_host, impl_->in_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    throw std::runtime_error("cudaMemcpy 输入失败");
  auto t1 = clock::now();

  // ---- 推理 ----
  if (!impl_->context->enqueueV2(impl_->bindings.data(), impl_->stream, nullptr))
    throw std::runtime_error("enqueueV2 失败");
  if (cudaStreamSynchronize(impl_->stream) != cudaSuccess)
    throw std::runtime_error("cudaStreamSynchronize 失败");
  if (cudaMemcpy(impl_->host_out.data(), impl_->dev_out, impl_->out_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    throw std::runtime_error("cudaMemcpy 输出失败");
  auto t2 = clock::now();

  // ---- 解码（列布局见 docs/0526-model-verification.md §1）----
  const float * out = nullptr;
  if (impl_->out_dtype == nvinfer1::DataType::kFLOAT) {
    out = (const float *)impl_->host_out.data();
  } else if (impl_->out_dtype == nvinfer1::DataType::kHALF) {
    cv::Mat half(impl_->out_rows, impl_->out_cols, CV_16F, impl_->host_out.data());
    half.convertTo(impl_->out_f32, CV_32F);
    out = impl_->out_f32.ptr<float>();
  } else {
    throw std::runtime_error("不支持的输出类型（只支持 fp32/fp16）");
  }

  struct Cand
  {
    int color_id, num_id;
    float conf;
    std::vector<cv::Point2f> points;
  };
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<Cand> cands;

  for (int r = 0; r < impl_->out_rows; r++) {
    const float * row = out + (std::size_t)r * impl_->out_cols;
    const float conf = sigmoid(row[8]);
    if (conf < config_.min_confidence) continue;

    Cand c;
    c.conf = conf;
    c.color_id = argmax(row + 9, 4);
    c.num_id = argmax(row + 13, 9);
    // 关键点重排（同济/深大同款）：(0,1)(6,7)(4,5)(2,3) → 左上,右上,右下,左下
    c.points = {
      {row[0] / (float)scale_, row[1] / (float)scale_},
      {row[6] / (float)scale_, row[7] / (float)scale_},
      {row[4] / (float)scale_, row[5] / (float)scale_},
      {row[2] / (float)scale_, row[3] / (float)scale_}};
    boxes.push_back(cv::boundingRect(c.points));
    scores.push_back(conf);
    cands.push_back(std::move(c));
  }

  if (!cands.empty()) {
    std::vector<int> keep;
    cv::dnn::NMSBoxes(
      boxes, scores, (float)config_.min_confidence, (float)config_.nms_threshold, keep);
    for (int k : keep) {
      const auto & c = cands[k];
      Armor armor(c.color_id, c.num_id, c.conf, boxes[k], c.points, config_.big_ratio_threshold);
      if (armor.name == ArmorName::not_armor) continue;  // 对齐同济 check_name
      armor.center_norm = {armor.center.x / img.cols, armor.center.y / img.rows};
      armors.push_back(std::move(armor));
    }
  }

  auto t3 = clock::now();
  timings_ = {delta_ms(t0, t1), delta_ms(t1, t2), delta_ms(t2, t3)};
  return armors;
}

YoloConfig yolo_config_from_yaml(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  if (!yaml["yolo"]) throw std::runtime_error("yaml 缺少 yolo: 检测配置块");

  auto y = yaml["yolo"];
  YoloConfig config;
  config.engine_path = tools::read_or<std::string>(y, "engine_path", config.engine_path);
  config.min_confidence = tools::read_or<double>(y, "min_confidence", config.min_confidence);
  config.nms_threshold = tools::read_or<double>(y, "nms_threshold", config.nms_threshold);
  config.big_ratio_threshold =
    tools::read_or<double>(y, "big_ratio_threshold", config.big_ratio_threshold);
  return config;
}

}  // namespace auto_aim
