# sp_vision_25 文件结构与线程架构（实读版）

> 依据：`reference/sp_vision_25` @ `bd9f5e7` **逐文件实读**（非二手描述）。目的：定下我方 rm_vision 的目录骨架与线程模型。
> 与 README"sp_vision_25 架构速览"互补：那节讲性质与对照，本文讲**结构、线程、同步原语与移植对照**。
> 最后更新：2026-10-08

---

## 一、总览：三层 + 两个辅助目录 + 应用层

| 目录 | 角色 | 构建产物 |
|---|---|---|
| `tools/` | 工具层：日志/队列/数学/标定辅助/调试 | `libtools`（STATIC） |
| `io/` | 硬件抽象层：相机/下位机/串口/IMU | `libio`（STATIC） |
| `tasks/` | 功能层：auto_aim / auto_buff / omniperception | `auto_aim`（**OBJECT**）等 |
| `src/` | 应用层：**每个兵种一个 main** | `standard` / `mt_standard` / `standard_mpc` / `uav` / `sentry` … |
| `calibration/` | 标定程序（各自独立可执行） | `capture` / `calibrate_camera` / `calibrate_handeye` / `calibrate_robotworld_handeye` / `split_video` |
| `tests/` | **每模块一个独立测试程序**（约 20 个） | `camera_test` / `cboard_test` / `minimum_vision_system` … |
| `configs/` | **一机一个 yaml**（兵种 + 相机 + 该机全部参数） | — |
| `assets/` | 模型权重与 demo 素材 | — |

核心设计决策（readme §3.4 + main 实读确认）：**程序只有一个 main**（相机独占，多程序会抢相机）；自瞄/打符等功能组按 `cboard.mode`（电控发来的挡位）在同一个 main 里切换。CMake 每层一个 `CMakeLists.txt`，应用/标定/测试各自 `add_executable` 后链接 `io auto_aim auto_buff tools`。

### 各层文件清单（我方需要的部分）

**tools/**：`thread_safe_queue.hpp`（有界队列，见下）、`thread_pool.hpp`、`exiter.{hpp,cpp}`（SIGINT 旗标）、`logger`（spdlog 封装）、`math_tools`（eulers / delta_time / square）、`img_tools`、`yaml.hpp`（yaml-cpp 封装 `tools::load/read`）、`crc`、`pid`、`trajectory`（弹道解算）、`extended_kalman_filter`、`plotter`（TCP 发 JSON 到 `127.0.0.1:9870`，PlotJuggler 画曲线）、`recorder`（视频+四元数，类 rosbag）

**io/**：
- `camera.hpp/cpp` — `CameraBase` 抽象（纯虚 `read(img, timestamp)`）+ `Camera` 工厂：按 yaml `camera_name` 选 `hikrobot` / `mindvision`；工厂里就读 `exposure_ms/gain/vid_pid`
- `cboard.hpp/cpp` — 下位机（SocketCAN）：公开 `bullet_speed`、`mode`、`imu_at(t)`、`send(Command)`
- `socketcan.hpp` — header-only 的 CAN 封装（epoll + 接收线程 + 守护线程）
- `hikrobot/ mindvision/ usbcamera/` — 相机驱动（**每个驱动内部自带 捕获线程 + 守护线程**）
- `serial/ gimbal/ dm_imu/ ros2/` — 串口库、云台调试、达妙 IMU、哨兵 ROS2（我方暂不需要）

**tasks/auto_aim/**（移植主体）：
- `armor.hpp` — `Armor` / `Lightbar` 数据结构；`Armor` 自带解算结果字段（`xyz_in_gimbal/world`、`ypr_in_gimbal/world`、`ypd_in_world`）+ 类型/颜色/优先级枚举
- `yolo.{hpp,cpp}` + `yolos/{yolov5,yolov8,yolo11}` — NN 检测工厂；`YOLOBase` 抽象 `detect()` / `postprocess()`
- `detector.{hpp,cpp}` — **传统灯条法**（阈值分割/几何校验/PCA 角点修正）；`yolov5.cpp:184` 里 `use_traditional_` 时对每块 NN 装甲板再跑一次传统法做二次确认
- `classifier` — 数字图案分类（小 ResNet）
- `solver.{hpp,cpp}` — PnP + yaw 优化（SJTU_cost）+ `set_R_gimbal2world()` + `reproject_armor()`
- `tracker.{hpp,cpp}` — 状态机（`detect_count` / `temp_lost_count` 门限）+ 单目标管理
- `target.{hpp,cpp}` — 整车 EKF
- `aimer.{hpp,cpp}` — 瞄准点决策（`comming_angle/leaving_angle/decision_speed` + yaw/pitch offset + 时间补偿）
- `shooter.{hpp,cpp}` — 开火决策（角度容差 + 距离判断）
- `voter` — 多帧投票（颜色/名字/类型）
- `multithread/mt_detector` — 异步推理队列（OpenVINO InferRequest）
- `multithread/commandgener` — **决策线程**（500Hz）
- `planner/`（+`tinympc/`）— MPC 轨迹规划（`standard_mpc` 用，可选）

---

## 二、线程架构（以 `src/mt_standard.cpp` 为蓝本）

### 线程清单

| # | 线程 | 创建处 | 职责 |
|---|---|---|---|
| 1 | **主线程** | main | 模式机；等检测结果 → `Tracker.track` → 把 targets 推给决策线程；打符模式时自己在主线程跑同步链路 |
| 2 | **detect_thread** | `mt_standard.cpp:64` | 取帧 → 预处理（resize 640×640 灰边）→ 启动异步推理 → 入队 |
| 3 | **CommandGener 线程** | `commandgener.cpp:15` | ~500Hz（2ms 循环）独立跑 `Aimer.aim + Shooter.shoot + cboard.send` |
| 4 | **相机捕获线程**（驱动内部） | `hikrobot.cpp:98` | 连续取流 → 推入容量 **1** 的队列（只留最新帧）；`read()` = 队列 pop（阻塞） |
| 5 | **相机守护线程**（驱动内部） | `hikrobot.cpp:17` | 100ms 周期健康检查，异常则 `reset_usb` 重开流 |
| 6 | **CAN 接收线程** | `socketcan.hpp:108` | epoll_wait(2ms) + 10µs 轮询 → `rx_handler` → `CBoard::callback` |
| 7 | **CAN 守护线程** | `socketcan.hpp:38` | 100ms 周期，链路断了自动重连 |
| 8 | OpenVINO 内部线程池 | 框架 | 真正执行异步推理（`THROUGHPUT` 模式多 stream），不在我们代码里 |

### 每环数据流与同步原语

```
相机捕获线程 ──queue(1,只留最新)──> detect_thread ──resize+start_async──> queue(16) ──> 主线程
                                                                          {img, t, InferRequest}
主线程 wait()+后处理 → tracker.track(armors, t) → CommandGener.push(targets, t, ...) 
                                                        │ 单槽(mutex+cv, >0.2s 判陈旧)
CommandGener 线程(500Hz) → aimer.aim() + shooter.shoot() → cboard.send(Command) ──CAN 8B──> 下位机
CAN 接收线程 → CBoard::callback → 四元数带时间戳 push ──queue(5000)──> imu_at(t帧-1ms) slerp 插值 ──> 主线程每帧取姿态
```

要点（都是刻意设计，移植时照抄）：

1. **相机驱动内建"最新帧"队列（容量 1）**：处理慢时旧帧直接丢，`read()` 永远拿最新帧，不堆积（与我方台账 §5"取帧策略"一致）
2. **检测队列（容量 16）**：`ThreadSafeQueue` 满时调 `full_handler`（打日志）并**丢弃新帧**；模板参数 `PopWhenFull` 可切换为丢最旧
3. **图像-姿态时间对齐**：主线程用**帧时间戳**取姿态 `imu_at(t - 1ms)`——从 5000 深的姿态队列里定位到该时刻前后两包做 **slerp 插值**（`cboard.cpp:22-45`）；`-1ms` 是曝光中点补偿（当时曝光 2.5ms 的一半）
4. **决策与感知解耦**：CommandGener 独立线程 500Hz 发指令，感知掉帧/停顿不影响云台指令流（陈旧 >0.2s 才停发）
5. **异步推理流水**：detect_thread 只做 resize + `start_async()`，不等结果；主线程 `pop()` 时才对最早那帧 `wait()` → 推理与采集天然重叠
6. **守护线程兜底**：相机/CAN 各有 100ms 周期守护线程，断线自动重连——赛场上免重启
7. **单线程版 `standard.cpp` 对照**：一个 while 循环顺序执行 `read→imu→detect→track→aim→send`，是全链路最小实现（调试/对照用）

### 配置结构（`configs/standard3.yaml` 实读）

分块：`enemy_color`｜NN（`yolo_name` + 各模型路径 + `min_confidence` + `use_traditional`）｜ROI｜传统法 9 个阈值｜tracker 门限｜aimer offsets｜shooter 容差/`auto_fire`｜相机（`camera_name/exposure_ms/gain/vid_pid`）｜**标定**（`camera_matrix` 9 元组 + `distort_coeffs` 5 元组 + `R_gimbal2imubody` + **手眼 `R_camera2gimbal/t_camera2gimbal`**）｜cboard（canid/接口）｜planner（fire_thresh/MPC 权重）

---

## 三、移植对照表（同济 → 我方）

| 层 | 同济 | 我方 | 动作 |
|---|---|---|---|
| 相机 | 海康 MV-CS016-10UC，hikrobot SDK，驱动内建 capture+daemon | 大恒 MER-139-210U3C，Galaxy SDK，BayerRG8 | **重写** `GalaxyCamera : CameraBase`；capture 线程 + queue(1) + 守护重连的模式照抄；Bayer→BGR 放哪待实测（台账 A2） |
| 下位机 | SocketCAN can0（0x100 四元数 / 0x101 弹速+模式 / 0xff 指令），Command 8B 含 `horizon_distance` | 协议待定（台账 D11） | **接口照抄**（`mode/bullet_speed/imu_at/send`），底层按我们协议实现 |
| IMU | C 板内置 BMI088，经 CAN 送四元数 | 同（依赖下位机协议） | 接口照抄 |
| NN 执行 | OpenVINO IR（x86 CPU/GPU） | TensorRT FP16（aarch64），0526.onnx（输入 [1,3,640,640] FP16，预处理 letterbox 640 + RGB + /255 —— 已验证与同济解码逻辑逐行兼容，见 `docs/0526-model-verification.md`） | **重写**执行层（`mt_detector` 的队列换成 TRT context/stream）；解码/颜色/类别映射复用 |
| 传统法 | 灯条法 + `use_traditional` 二次复核 | 同（纯 OpenCV 算法） | 直接移植；阈值按我们的曝光/增益重调 |
| solver/tracker/target/aimer/shooter | 纯算法（Eigen/OpenCV） | 同 | **直接移植**；把 `camera_matrix/distort_coeffs` 换成我们的标定（`calibration/`）；`R_gimbal2imubody`、手眼 `R_camera2gimbal/t_camera2gimbal` 待做（台账 D11/标定） |
| 外挂 | auto_buff / omniperception / sentry / uav / ros2 / mpc planner | **只打步兵（3V3）** | 不移植（auto_buff 留作参考） |
| 标定 | `calibration/` 4 程序 | 内参已有（2026-08-15）；PnP 验证与手眼待做 | 移植 `calibrate_handeye`（我们也要 R_camera2gimbal） |
| 测试 | 每模块一个 + `minimum_vision_system` | 同思路 | **第一步照 `minimum_vision_system` 做最小闭环** |

---

## 四、建议的我方骨架（定稿待建）

```
rm_vision/
├── CMakeLists.txt          # 顶层：OpenCV/Eigen/yaml-cpp/spdlog/fmt/nlohmann + add_subdirectory(tools io tasks)
├── configs/
│   └── standard3.yaml      # 一机一档；相机块换成 galaxy 键（设备 SN/曝光/增益/像素格式/ROI）
├── io/
│   ├── camera.hpp/cpp      # CameraBase + 工厂（camera_name: galaxy）
│   └── galaxy/             # 大恒驱动（capture 线程 + queue(1) + 守护线程）
├── tasks/auto_aim/         # armor / detector / solver / tracker / target / aimer / shooter / voter
│   ├── yolo.hpp/cpp        # TRT 执行层（后处理复用）
│   └── multithread/        # mt_detector(TRT 版) / commandgener
├── tools/                  # thread_safe_queue / exiter / logger / math_tools / yaml / img_tools / plotter / recorder / ekf …
├── src/
│   └── standard.cpp        # 步兵 main：先单线程闭环，跑通后 mt 化
├── tests/
│   ├── camera_test.cpp     # 取流 + 参数设置 + 帧率实测（台账 A1 剩余）
│   └── minimum_loop.cpp    # 相机 + TRT + 解码存图，端到端计时（台账 B4）
├── calibration/            # 现有内参台账 + 后续手眼程序
├── models/  docs/  assets/
└── reference/              # （不入库）
```

**推进顺序**：骨架 CMake 能编译 → `camera_test`（取流+设参数+实测帧率）→ `minimum_loop`（+TRT 推理+存图，端到端计时）→ 移植 solver → tracker → mt 化 + commandgener。
