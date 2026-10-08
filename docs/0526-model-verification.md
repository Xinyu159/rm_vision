# 0526 模型验证报告

2026-10-08 ｜ 结论：**模型可用；接口与同济 Armor/YOLO 体系几乎逐行兼容。**

## 1. 模型规格（`models/0526.onnx`，深大 RP24 开源，2025-08-09 版）

- 网络：魔改 YOLOv5（MobileNetV3 backbone）；原环境 NUC + OpenVINO GPU 纯推理 ~100FPS
- 输入：`images` [1,3,640,640]，**float16**；RGB、除以 255（无 mean/std）
- 输出：`output` [1,25200,22]，float32，每行 22 列：

| 列 | 含义 | 备注 |
|---|---|---|
| 0-7 | 4 个关键点 (x,y)×4 | 左上角起逆时针 |
| 8 | objectness | 先 sigmoid 再比阈值 |
| 9-12 | 颜色 | **实测：0=蓝 1=红 2=灰(熄灭) 3=紫** |
| 13-21 | 编号 | 0=G 哨兵，1-5=一~五号，6=O 前哨站，7=Bs 基地小装甲，8=Bb 基地大装甲 |

⚠️ 深大 README 把颜色写作"红蓝灰紫"，其仓库内 `OpenvinoInfer.cpp`（2023 年文件）注释写 red:0/blue:1——**均与 0526 实测不符**，以本报告实测为准。

## 2. 验证方法与结果

- 素材：同济 demo 视频抽样 44 帧（1440×1080，`reference/sp_vision_25/assets/demo/demo.avi`）
- 方法：onnxruntime（CPU）+ letterbox 640 预处理 + 同济 `yolov5.cpp` 的解析逻辑；脚本 `tools/verify_0526.py`
- 结果：
  - ✅ 四点框精准（贴合灯条与装甲板上下沿），见 `docs/img/0526_demo_field.jpg`
  - ✅ 数字分类正确（场上 3、4 号机器人与其装甲板数字一致）
  - ✅ 熄灭灯条正确判为灰(2)
  - ✅ 颜色实测：蓝灯→索引 0、熄灭→2；用通道交换合成红灯→索引 1（素材中无实拍红灯，此条为合成验证）
  - 置信度 0.67~0.98；大特写 / 运动模糊帧偶有漏检，属正常范围

## 3. 与同济接口的映射（对照 `tasks/auto_aim/yolo.hpp`、`armor.hpp`、`yolos/yolov5.cpp`）

| 环节 | 同济实现 | 0526 适配 | 改动量 |
|---|---|---|---|
| 预处理 | letterbox 640、BGR→RGB、/255 | 一致 | 无 |
| 输出解析 | 22 列布局 + sigmoid(col8) + argmax 颜色/编号 | 一致 | 无 |
| 关键点重排 | `(0,1)(6,7)(4,5)(2,3)` → [左上,右上,右下,左下] | 一致 | 无 |
| Armor 构造 | `color = color_id==0?blue : color_id==1?red` | 0526 颜色索引恰好一致 | 无 |
| 编号映射 | 0→sentry、1-5→one..five、6→outpost、7→base | 一致 | 无 |
| NMS | NMSBoxes(四点外接框, conf) | 一致 | 无 |
| 推理后端 | OpenVINO IR | ONNX(f16) | 换 onnxruntime-gpu / TensorRT |

**必须处理的 3 个差异：**

1. **num=8**：0526 是 Bb 基地大装甲，同济构造函数把 8 当 not_armor → 映射时加一行 `8 → base + big`（3V3 步兵基本用不到）
2. **big/small type**：0526 不直接输出，需按关键点几何（`ratio = 长/宽`）或编号自行判定；同济 v5 构造里 `num_id==1 → big` 的写法与 0526 编号体系不符，**不要照搬**
3. **FP16 输入**：喂 ONNX 需要 float16（onnxruntime 手动 cast；TensorRT / OpenVINO 会自动处理）

## 4. 后续

下一步：步兵（3V3）单兵种架构设计——io 层接大恒相机（Galaxy SDK）、detector 层按本映射实现、推理后端选型（TensorRT vs onnxruntime-gpu）。
