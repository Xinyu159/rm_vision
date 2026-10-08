# rm_vision

RoboMaster 视觉开发工作区（Jetson 平台，aarch64 / JetPack 5.1.4）。

## 参考代码（reference/，各自独立 git 仓库，不纳入本仓库版本管理）
- `sp_vision_25` — 同济大学 SuperPower 战队 25 赛季视觉框架（算法思想参考）
- `RobotDetectionModel` — 深大 RobotPilots RP24 自瞄模型仓库，本工程使用其 `Model/0526.onnx`

## 本工程目标
参考 sp_vision_25 的框架结构，开发适配自有关卡的 RM 视觉代码：
- 检测模型：0526.onnx（与同济不同）
- 相机：大恒水星一代 MER-139-210U3C（USB3.0，与同济不同）

## 环境备忘
- 相机 SDK：~/sdk/galaxy/Galaxy_Linux-arm64_Gige-U3_2.4.2507.8231.zip（待安装）
- Jetson IP：10.42.0.80（网线直连笔记本，笔记本共享上网）
