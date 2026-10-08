# 结论与待定事项（传承台账）

> **用途**：上半部分是**已确认、可作为后续开发依据的结论**，下半部分是**必须弄清、否则会返工的问题**。
> 维护约定：有了新结论，就从"待弄清"挪到"已确认"；被推翻的结论写清纠正理由后并入"已纠正的误区"。
> 最后更新：2026-10-08

**一句话现状**：模型接口验证完毕；推理后端拍板 TensorRT FP16（NX 锁频后 164 FPS）；相机 SDK 已装但尚未摸参数；下一步 = C++ 最小闭环 + 相机参数摸底。

---

## 一、已确认结论

### 1. 平台（Jetson Xavier NX 8GB）

| 项 | 结论 |
|---|---|
| 硬件/系统 | Xavier NX Developer Kit 8GB；JetPack 5.1.4（L4T 35.6.4）/ Ubuntu 20.04 / aarch64；用户 `rm` |
| 连接 | 网线直连笔记本 `10.42.0.80`（NM 共享模式 + NAT）；Jetson 经反向隧道借用笔记本代理 `127.0.0.1:17897` |
| 内存 | 总量 6833MB、常态可用 2~4GB —— 建 engine 这类重活前杀 VS Code server 的 node 进程腾内存（效果显著） |
| **功耗模式（关键）** | **默认 MODE_10W_DESKTOP = 半血**（4 核、GPU 上限 510MHz）。已切 **MODE_15W_6CORE(ID=2) + `jetson_clocks`**（GPU 1109.25MHz、EMC 1600MHz、6 核在线）。**nvpmodel 模式重启保持，jetson_clocks 不保持** → 部署需开机脚本；20W 档（ID 6/7/8）可作后手（主要抬 CPU 频率上限）。恢复省电：`sudo nvpmodel -m 5` 或重启 |

### 2. 模型（0526.onnx）

- 结构 = MobileNetV3 backbone 版 YOLOv5；输入 **FP16 [1,3,640,640]，RGB、/255**
- 输出 [1,25200,22]：列 0-7 = 四点坐标（(0,1)(6,7)(4,5)(2,3) 对应 **TL,TR,BR,BL**）；列 8 = objectness(sigmoid)；列 9-12 = 颜色 argmax，**实测 [蓝0 红1 灰2 紫3]**（深大 README 文字/旧代码有出入，**以实测为准**，与同济 `Armor(0→蓝,1→红)` 一致）；列 13-21 = 数字/类型（0=G、1-5、6=O、7=Bs、8=Bb）
- 与同济 Armor 接口**语义逐行兼容** → 检测层可直接套同济的 Armor 数据结构；移植注意 3 处差异：① num=8 时 base 归一别当大装甲 ② 大/小装甲类型需自判（模型不输出 type） ③ 输入是 FP16
- 完整验证过程见 `docs/0526-model-verification.md`

### 3. 推理后端（拍板：TensorRT FP16 直连，C++）

- **不用 INT8**：角点抖动风险（轨迹预测对噪声敏感）；FP16 精度稳
- **不用 onnxruntime**：Jetson aarch64 无官方 wheel；CUDA EP 慢（1.5~3×）；TRT EP 只是包装层、同样要建 engine 还多一层开销 → 性能不可能超过直连
- engine：`models/0526_fp16.engine`（~5.6MB，gitignore，**与设备 + TRT 版本绑定**）；`0526.timingcache` 加速重建；一次性构建 ~14.5min（受内存影响，**必须限制 `--workspace=1024`**，否则反复跳过 tactic 卡死）；加载/反序列化 ~2.2s；**确定性**（同输入重复推理结果一致，适合跑分/回归）
- **实测（15W+锁频，640×640 FP16）**：GPU 单次 **6.08ms**、p99 6.13ms、吞吐 **164.1 FPS**；同款在省电档只有 10.41ms/95.8FPS
- 抖动语义："0.08ms" = 单次 10s 基准内 **GPU 内核时间**的 max−min（省电档）；锁频后为 6.04~6.21ms。**这是 GPU 时间、不是端到端** —— 端到端抖动待 C++ 闭环实测

### 4. 帧率预算（关键账）

| 目标 | 每帧预算 | 结论 |
|---|---|---|
| 89 fps | 11.24 ms | 全链路预估 7.5~8.5ms（推理 6.08 + 拷贝 0.4 + 预处理 0.3-0.5 + 后处理 0.5-1 + PnP/EKF 0.3）→ **有余量** |
| 180 fps | 5.56 ms | **NN 单独 6.08ms 已超 → 直连 NN 路径到不了 180**；正主是传统视觉（灯条法，目标 <5ms），或"传统 180 + NN 低频复核"双频混合 |

### 5. 相机（大恒 MER-139-210U3C）

- USB3、VID:PID `2ba2:4d55`、lsusb 名 `Daheng Imaging MER-139-210U3C`；Galaxy ARM64 SDK（2.4.2507.8231）**已装**（库在 `/lib`、udev 规则在 `/etc/udev/rules.d/99-galaxy-u3v.rules`、头文件/样例在 `~/sdk/galaxy/.../Galaxy_camera/{inc,lib/armv8,sample}`）
- 同系列 131-210U3C 公开规格：1280×1024、**最高 210fps**、1/2" PYTHON1300 **全局快门**、4.8µm；本型号精确上限公开查不到 → **待 SDK 直读**（见待办 A1）
- **帧率公式（手册原文）**：`T_f = Max(ImageSize×10⁶/BandWidth, ImageSize×10⁶/ThroughputLimit, T_acq, T_exp)`，`ImageSize = W×H×字节数+84`（8bit=1、10/12bit=2）→ 拉帧率四旋钮：**8bit 像素格式（BayerRG8）**、**ROI 裁剪**、**缩短曝光**、**AcquisitionFrameRate(先 Enable) + DeviceLinkThroughputLimit 限流检查**
- 12bit 下 1280×1024@210fps 需 550MB/s > USB3 理论 400MB/s → **8bit（275MB/s）才谈得上满帧率**
- 全局快门 → 高速旋转无果冻效应（RM 场景加分）
- 取帧策略（设计决定）：**永远取最新帧**，处理追不上就丢旧帧，不排队堆积

### 6. 工程与开发流程

- 工程：Jetson `~/rm_vision`（git/main）；参考 `reference/sp_vision_25`（bd9f5e7，架构思想）、`reference/RobotDetectionModel`（0526 出处，另有 0708）
- 平台差异三大件：**aarch64（非 x86）、大恒相机（非同济海康）、0526.onnx + TensorRT（非 OpenVINO）** → io 层与推理层是重写重点；解算/跟踪/规划等算法思想可从同济移植
- 开发链路：VS Code Remote-SSH（`ssh jetson`）；文档集中在 `docs/`；md 预览装 Markdown Preview Enhanced（或内置 Ctrl+Shift+V）
- 下载约定：**所有下载由用户本人完成**（只需给出名字/链接）；给 Jetson 的大文件走"笔记本下载 → 局域网 scp → Jetson 解压"（别用慢速隧道硬拉）
- 网络：校园网直连常不通，**别用 ping 判断**（封 ICMP），用 `curl -I`；代理在笔记本 `127.0.0.1:17897`（http/socks5 同端口）
- 运维坑（已踩）：`pkill -f` 会匹配到自己 ssh 脚本正文导致自杀 → 用 `pkill -x` 或字符串拼接；TRT 建引擎必须限 `--workspace` + 先腾内存

### 7. 已纠正的误区（别再犯）

- ❌ "相机只有 60fps 级" → 同系列上限 **210fps**（全局快门），且 SDK 有完整提帧手段
- ❌ "深大颜色映射写错了" → 模型实测映射与同济一致，是深大 README 文字/旧代码表述有出入
- ❌ "省电档也能测性能" → 默认 10W 档 GPU 被锁 510MHz，**任何性能结论必须在锁频态下测**
- ❌ "onnxruntime 是更快的另一个选项" → 同层封装：TRT-EP 持平或更慢，CUDA-EP 明确更慢

---

## 二、必须弄清的问题（开放项）

**A. 相机（最先做，影响全链路设计）**
1. 用 SDK 直读本机真实参数：分辨率上限、`AcquisitionFrameRateRange`、支持的 PixelFormat、ROI 步进、`DeviceLinkThroughputLimit` 范围；并**实测**：8bit 全幅最高帧率、典型 ROI 档帧率、1ms 曝光下能跑到多少
2. Bayer→RGB 转换放在哪：CPU(OpenCV) / GPU(自写 CUDA 或 NPP) / SDK 输出 —— 实测 1.3MP 各自耗时后定
3. 相机时间戳质量（SDK 帧时间戳 vs 主机时间）→ 决定 EKF 的时间基准

**B. 性能（决定 89/180 能否落地）**
4. C++ 闭环**端到端耗时分解**（预处理/后处理/PnP 在 NX 的真实耗时）—— 目前只有 GPU 推理一环有实测
5. 传统视觉 180fps 可行性：灯条法原型在 NX CPU 上能否 <5ms；"传统 + NN 双频混合"方案是否成立
6. DLA（双核 NVDLA）能否卸载 0526 部分算子、收益多少 —— 低优先级

**C. 工程接口（开工闭环前定）**
7. 线程模型与零拷贝方案：取帧/推理/后处理线程分工；pinned 内存 + GPU 预处理；时间戳如何贯穿到 EKF
8. 配置面：相机 yaml（fps/ROI/曝光/gain/PixelFormat）+ 模型路径 + 阈值参数（对齐同济 configs 风格）
9. **标定内参与 ROI 的兼容性**：已有内参是在什么分辨率/ROI 下标定的？改 ROI 后内参要重标或换算 —— 动 ROI 前必须先弄清
10. 只打步兵（3V3）→ 后处理类别裁剪策略（丢弃非目标类别，阈值/NMS 是否需相应调整）

**D. 后续阶段（未开始）**
11. 下位机通信协议（串口/网口 + 帧格式）与裁判系统数据接入
12. 部署化：开机脚本（nvpmodel + jetson_clocks）、进程守护（systemd）、engine 构建脚本化（换设备/升级 TRT 时一键重建）

---

## 三、建议推进顺序

1. **相机参数摸底**（A1~A3）—— 当天可完成
2. **C++ 最小闭环**：engine 加载 + 相机取帧 + 推理 + 解码存图 → 量化端到端耗时（顺带回答 C7/C8 的初版）
3. **传统法原型**验证 180fps 可行性（B5）
4. 架构铺开（对齐同济 `tasks/auto_aim`、`io/`、`tools/`、`configs/` 目录）
