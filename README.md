# rm_vision

RoboMaster 视觉开发工作区（Jetson 平台，aarch64 / JetPack 5.1.4 / Ubuntu 20.04，用户 rm）。

## 本工程目标

参考 sp_vision_25（同济大学 SuperPower 战队 25 赛季视觉开源）的框架结构与算法思想，开发适配自有关卡的 RM 视觉代码。算法链路思想可复用（检测 → PnP 解算 → EKF 跟踪 → 弹道/轨迹规划 → 开火决策），但平台、相机、模型均不同：

| 项 | 同济 sp_vision_25 | 本工程 |
|---|---|---|
| 运算平台 | x86 NUC（i7-1260P）/ Ubuntu 22.04 | Jetson（aarch64）/ Ubuntu 20.04 |
| 相机 | 海康 MV-CS016-10UC（USB3） | 大恒水星一代 MER-139-210U3C（USB3） |
| 检测模型 | YOLOv5/v8/v11，OpenVINO IR（.xml+.bin） | `models/0526.onnx`（深大 RP24 开源） |
| 推理后端 | OpenVINO 2024.6 | TensorRT FP16（JetPack 5.1.4 自带；`0526.onnx` → `models/0526_fp16.engine`，trtexec 构建、**设备绑定**，换设备/升级 TRT 需重建） |
| 下位机 | RM 开发板 C 型（BMI088 + MicroUSB 虚拟串口） | 待定 |

## 目录

- `models/` — 检测模型本体入库，与代码版本对齐（`0526.onnx`，4.2MB）
- `docs/` — 分析与验证文档
- `io/` — 硬件 IO 层：`camera.hpp` 相机抽象 + `galaxy/` 大恒驱动（捕获线程 + 守护重连线程，容量 1 丢旧留新队列）
- `tasks/` — 算法层：`auto_aim/` 自瞄（`armor` 装甲板数据结构（语义对齐同济）、`yolo` TensorRT 检测后端：预处理/推理/解码/NMS）
- `tools/` — C++ 通用件（logger / exiter / math_tools / thread_safe_queue / yaml / mjpeg_server）+ 工具脚本（`gx_dump.py` 相机参数复读、`verify_0526.py` python 参考实现、`cross_check.py` C++↔python 逐框对拍）
- `configs/` — 机器人配置（`standard3.yaml`：相机参数按实测写全 + 标定内参 + `yolo:` 检测参数）
- `src/` — 主程序（`standard.cpp` 空壳，随阶段推进长出检测/跟踪/决策）
- `tests/` — 模块测试（`camera_test` 取流/帧率实测；`camera_view` 实时预览；`detect_view` 检测预览：相机→检测→画框→浏览器；`yolo_test` 单图离线检测/耗时对拍）+ `cli_common.hpp` 公共命令行件
- `calibration/` — 相机标定参数（内参 yaml + 标定记录表）
- `reference/` — 参考仓库（各自独立 git，本仓库不追踪，详见下节）

## 构建与运行

```bash
mkdir -p build && cd build && cmake .. && make -j4
./build/standard                                          # 主程序空壳（取流 + 帧率打印）
./build/camera_test --seconds=3                           # 相机取流/帧率实测
./build/camera_test --exposure=4000 --roi=640x512 --decimation=2 --save=frame.png
./build/camera_view --serve=8080 --scale=0.75             # 实时预览（日常）：浏览器开 http://10.42.0.80:8080/
./build/camera_view                                       # 或在 Jetson 本机显示器开窗口（需 DISPLAY；无显示器用 --serve）
./build/detect_view --serve=8080 --scale=0.75             # 检测预览：相机→0526 检测→画框/标签→浏览器
./build/yolo_test docs/img/0526_demo_field.jpg            # 单图离线检测：逐框打印 + 耗时分位数（和 python 参考对拍用）
```

- 依赖：OpenCV 4.5.4 / Eigen3 / yaml-cpp / spdlog / fmt + TensorRT/CUDA（JetPack 自带 nvinfer/cudart）+ 大恒 Galaxy ARM64 SDK（头文件路径见顶层 `CMakeLists.txt` 的 `GALAXY_INC`）
- `detect_view`/`yolo_test` 需在**仓库根目录**运行（配置里模型路径是相对路径），且要 `models/0526_fp16.engine` 在位
- 注意：OpenCV 参数解析器**必须用 `=` 传值**（`--exposure=4000`）；写空格分隔会把数值当成配置文件路径
- 相机设置会跨开关机保留，所以 `configs/standard3.yaml` 里几何参数（ROI/抽行）写全，保证每次启动形态确定
- **预览叠加文字只能 ASCII**：`cv::putText` 的 Hershey 字体渲染不了中文（会显示成"??"），改叠加内容时注意
- 白平衡（`balance_white_auto`）：默认 `Once` = 开流后按**当时画面**算一次；若这一刻镜头前恰好是红/蓝色物体，补偿会被带下去（实测踩过：正常画面变蓝调）→ 窗口模式按 `w` 重做，或加 `--wb=Off` 沿用当前增益不再重算
- MJPEG 预览的 JPEG 编码在**服务线程**做（`tools/mjpeg_server.hpp`），主循环只画+发布——编码一次 ~15ms，放主循环会把检测管线从 ~50fps 拖到 27fps（实测；耗时分解见台账 §6/B4）
- 相机是**独占资源**：同机同时只能有一个程序开着相机（先退旧进程再起新进程），端口 8080 同理
- **逐框对拍**（校验 C++ 解码与 python 参考一致）：先在 Jetson 收集 `./build/yolo_test 图... > cpp.txt 2>&1`，再在装了 onnxruntime 的机器上 `MODEL=... python3 tools/cross_check.py cpp.txt 图...`（差异 ≤0.03 阈值边缘自动判定；2026-10-09 通过：7 图 Δconf≤0.001、角点≤0.6px，1 例阈值边缘）

## 文档

- [0526 模型验证报告](docs/0526-model-verification.md) — 四点装甲板检测确认可用 + 与同济 Armor/YOLO 接口映射（第一步结论）
- [结论与待定事项](docs/conclusions-and-open-items.md) — 已确认结论台账 + 必须弄清的问题清单（传承用，持续更新）
- [sp_vision_25 文件结构与线程架构](docs/sp-vision-25-structure-and-threads.md) — 实读版：三层结构 / 线程清单 / 同步原语 / 移植对照表

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

**本工程落地时的改造点**：① ~~相机按 `io/camera.hpp` 抽象新增大恒实现~~ ✅ 已完成（`io/galaxy/`，Galaxy SDK C++，2026-10-08 编译 + 跑流实测通过）；② ~~模型链路换成 0526.onnx + TensorRT~~ ✅ 已完成（`tasks/auto_aim/yolo`，TRT FP16，检测闭环 2026-10-08 实测通过；与同济的差异：未做软件 ROI、传统检测器 `detector` 延后到传统法阶段）；③ 下位机协议待定。

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
