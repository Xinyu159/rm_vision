# rm_vision

RoboMaster 视觉开发工作区（Jetson 平台，aarch64 / JetPack 5.1.4 / Ubuntu 20.04，用户 rm）。

## 本工程目标

参考 sp_vision_25（同济大学 SuperPower 战队 25 赛季视觉开源）的框架结构与算法思想，开发适配自有关卡的 RM 视觉代码。算法链路思想可复用（检测 → PnP 解算 → EKF 跟踪 → 弹道/轨迹规划 → 开火决策），但平台、相机、模型均不同：

| 项 | 同济 sp_vision_25 | 本工程 |
|---|---|---|
| 运算平台 | x86 NUC（i7-1260P）/ Ubuntu 22.04 | Jetson（aarch64）/ Ubuntu 20.04 |
| 相机 | 海康 MV-CS016-10UC（USB3） | 大恒水星一代 MER-139-210U3C（USB3） |
| 检测模型 | YOLOv5/v8/v11，OpenVINO IR（.xml+.bin） | `models/0526.onnx`（深大 RP24 开源） |
| 推理后端 | OpenVINO 2024.6 | 待定：Jetson 上建议 TensorRT / onnxruntime-gpu，ONNX 需转换 |
| 下位机 | RM 开发板 C 型（BMI088 + MicroUSB 虚拟串口） | 待定 |

## 目录

- `models/` — 检测模型本体入库，与代码版本对齐（`0526.onnx`，4.2MB）
- `docs/` — 分析与验证文档
- `tools/` — 工具脚本
- `reference/` — 参考仓库（各自独立 git，本仓库不追踪，详见下节）

## 文档

- [0526 模型验证报告](docs/0526-model-verification.md) — 四点装甲板检测确认可用 + 与同济 Armor/YOLO 接口映射（第一步结论）

## 参考代码（reference/，不纳入本仓库版本管理）

- `sp_vision_25` — 同济 SuperPower 25 赛季视觉框架（main 分支，参考版本 `bd9f5e7`，2026-08-13）
- `RobotDetectionModel` — 深大 RobotPilots RP24 自瞄模型仓库（本工程 0526 模型出处，另有 0708）

### sp_vision_25 架构速览

- **性质**：C++17 / CMake。入口在 `src/*.cpp`，按兵种分（standard / sentry / uav），含多线程版（`mt_*`）与各类 debug 版
- **tasks/**：`auto_aim/`（`armor` 装甲板、`detector`+`yolo`+`yolos/` 检测、`solver` PnP 解算、`tracker` EKF 跟踪、`aimer`/`shooter`/`voter`/`classifier`、`planner/` 轨迹规划）、`auto_buff/`（能量机关，`yolo11_buff`）、`omniperception/`（全向感知）
- **io/**：`camera.hpp` 相机抽象基类（`CameraBase::read` 接口）→ 厂商实现 `hikrobot/`（海康）、`mindvision/`（迈德威视）、`usbcamera/`；另有 `cboard`（裁判系统通信）、`serial/`、`socketcan.hpp`、`gimbal/`、`dm_imu/`（达妙 IMU）、`ros2/`
- **tools/**：EKF、PID、trajectory、plotter、recorder、logger、img_tools、thread_pool 等通用件
- **configs/**：按兵种 YAML（camera.yaml 含相机参数）；**assets/**：OpenVINO 模型；**calibration/**：相机/手眼标定工具源码；**tests/**：各模块测试程序
- **依赖**：OpenCV、Eigen3、fmt、spdlog、yaml-cpp、nlohmann_json、OpenVINO 2024.6（auto_buff 另用 Ceres）

**本工程落地时的改造点**：① 相机按 `io/camera.hpp` 抽象新增大恒实现（Galaxy SDK C++ 或 gxipy）；② 模型链路换成 0526.onnx（Jetson 推理后端待选型）。

## 相机 SDK

大恒 Galaxy ARM64 SDK（Gige+U3，2.4.2507.8231）**已安装并验证**（2026-10-08）：

- 运行库在 `/lib/`：`libgxiapi.so`、`liblog4cplus_gx.so`、GenTL 生产者 `GxGVTL.cti`/`GxU3VTL.cti`（已在 ldconfig 中）
- udev 规则 `/etc/udev/rules.d/99-galaxy-u3v.rules`（VID `2ba2` → 0666 + uaccess）
- 相机已在 USB3 总线上枚举：`lsusb -d 2ba2:4d55` → `Daheng Imaging MER-139-210U3C`
- 开发用头文件/库/示例在解压包内 `~/sdk/galaxy/Galaxy_Linux-arm64_Gige-U3_2.4.2507.8231/Galaxy_Linux-arm64_Gige-U3_2.4.2507.8231/Galaxy_camera/`：`inc/`（GxIAPI.h、DxImageProc.h）、`lib/armv8/`、`sample/GxSingleCamColor/`（彩色采集示例，io 层参考）
- Python 绑定 gxipy 未装（C++ 链路不需要）

## 环境备忘

- Jetson：`10.42.0.80`（网线直连笔记本 `enxec1ac3028219`，笔记本 NM 共享模式 + NAT 供其上网）
- 远程开发：VS Code Remote-SSH（`ssh jetson`）
- 网络：校园网直连常不通，走笔记本代理 `127.0.0.1:17897`（Jetson 侧经 `ssh -R` 反向隧道借用）
