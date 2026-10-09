#include "auto_aim/yolo.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
  void * host_in = nullptr;  // 输入主机缓冲（优先 pinned，见 ctor）
  bool host_in_pinned = false;
  std::vector<void *> bindings;
  std::vector<std::uint8_t> host_out;

  // CPU 侧缓存（避免每帧重新分配）
  cv::Mat bayer_bgr, canvas, out_f32;
  std::vector<cv::Mat> channels;   // split 输出缓存（B G R）
  int last_nw = -1, last_nh = -1;  // 上次 letterbox 实际尺寸（黑边只在变化时重清）

  ~Impl()
  {
    if (dev_in) cudaFree(dev_in);
    if (dev_out) cudaFree(dev_out);
    if (host_in) {
      if (host_in_pinned)
        cudaFreeHost(host_in);
      else
        std::free(host_in);
    }
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

  // 输入主机缓冲用 pinned：H2D 实测 0.54ms → 0.19ms（2.4MB fp16）；失败则退回普通内存
  if (cudaHostAlloc(&impl_->host_in, impl_->in_bytes, cudaHostAllocDefault) == cudaSuccess) {
    impl_->host_in_pinned = true;
  } else {
    impl_->host_in = std::malloc(impl_->in_bytes);
    if (!impl_->host_in) throw std::runtime_error("分配输入主机缓冲失败");
    tools::logger()->warn("[yolo] pinned 主机内存分配失败，退回普通内存（H2D 会慢些）");
  }

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
  // 2026-10-09 提速（离线分环节实测，见台账 §6/B4；①②③输出与旧实现逐位一致，已校验）：
  //   ① resize 直接写 canvas 左上 ROI —— 免临时 Mat + copyTo
  //   ② 黑边只在尺寸变化时清一次 —— 输入尺寸固定时稳态零开销
  //   ③ 尾段融合：split + 按平面 convertTo(1/255, 输入类型) 直写主机缓冲，
  //      免 fp32 NCHW 中转（原 blobFromImage+convertTo 约 2.8ms → 约 0.8ms）
  //   ④ H2D 走 pinned + 异步（约 0.54 → 0.19ms；与推理同流，天然先拷后推）
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

  if (impl_->canvas.rows != impl_->input_h || impl_->canvas.cols != impl_->input_w ||
      impl_->canvas.type() != CV_8UC3) {
    impl_->canvas.create(impl_->input_h, impl_->input_w, CV_8UC3);
    impl_->last_nw = impl_->last_nh = -1;
  }
  if (nw != impl_->last_nw || nh != impl_->last_nh) {  // 黑边区域变了才重清
    impl_->canvas.setTo(cv::Scalar(0, 0, 0));
    impl_->last_nw = nw;
    impl_->last_nh = nh;
  }
  cv::resize(src, impl_->canvas(cv::Rect(0, 0, nw, nh)), cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

  // BGR→RGB、/255、拼成 NCHW（平面序 R|G|B，与 blobFromImage(swapRB=true) 输出逐位一致）
  const int in_cv_type = impl_->in_dtype == nvinfer1::DataType::kHALF  ? CV_16F
                         : impl_->in_dtype == nvinfer1::DataType::kFLOAT ? CV_32F
                                                                         : -1;
  if (in_cv_type < 0) throw std::runtime_error("不支持的输入类型（只支持 fp32/fp16）");
  cv::Mat host_in(
    impl_->input_h * 3, impl_->input_w, in_cv_type, impl_->host_in);  // 覆盖主机缓冲的视图
  cv::split(impl_->canvas, impl_->channels);                          // B G R
  impl_->channels[2].convertTo(host_in.rowRange(0, impl_->input_h), in_cv_type, 1.0 / 255.0);
  impl_->channels[1].convertTo(
    host_in.rowRange(impl_->input_h, 2 * impl_->input_h), in_cv_type, 1.0 / 255.0);
  impl_->channels[0].convertTo(
    host_in.rowRange(2 * impl_->input_h, 3 * impl_->input_h), in_cv_type, 1.0 / 255.0);

  if (cudaMemcpyAsync(
        impl_->dev_in, impl_->host_in, impl_->in_bytes, cudaMemcpyHostToDevice, impl_->stream) !=
      cudaSuccess)
    throw std::runtime_error("cudaMemcpyAsync 输入失败");
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
