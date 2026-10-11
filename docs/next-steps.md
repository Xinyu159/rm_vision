# 下一步计划（2026-10-11 更新）

> 用途：解算（PnP）阶段收口后的开发路线。
> **A = 需要你（用户）做的事**（不用写代码，几十分钟量级，很多现在就能做）；
> B = 我这边的开发顺序；C = 上实车当天的清单；D = 现有命令速查。
> 联动的账本：`docs/conclusions-and-open-items.md`（结论与开放项）。

## 0. 现在具备的能力（对照用）

| 能力 | 状态 |
|---|---|
| 检测（相机 → 0526 → 四点框） | ✅ 单帧 ≈11ms（≈92fps 管线），`detect_view` 浏览器看 |
| 解算（四点 → 三维位姿：距离/yaw） | ✅ 单块 p99 0.253ms（锁频态），`pnp_view` 看 `d=`/`yaw=`/重投影对齐/`err` |
| 大/小装甲型别 | ✅ 今年无大板 → 全部按小处理，误判为零（练车有旧大板再开几何判定） |
| 下位机串口（'SP' 帧四元数） | ✅ 代码 + 离线自检全绿（`serial_test`）；实车 live 联调待板接 ttyACM0 |
| 标定采集 / 手眼解算工具 | ✅ `capture`（浏览器可控制）+ `calibrate_robotworld_handeye` + `pnp_view`（2026-10-11 就绪，等实车采集） |
| 跨帧跟踪（速度/角速度） | ❌ 还没有 tracker（B1） |
| 瞄准/弹道 | ❌ 还没有 aimer（B4） |
| 手眼绝对姿态 | ❌ 外参还是"相机理想正装"占位值（A5 标定后才有）；`pnp_view` 画面会红字提示 EXTR NOMINAL |

## A. 需要你做的事（按先后；A1 不用相机，现在就能做）

**A1. 量手里的装甲板（2 分钟）** —— 程序里写死三个常量（`tasks/auto_aim/solver.cpp` 顶部），量完对一下：
- ① 两灯条**中心线**间距（现按 **135.0 mm**）
- ② 灯条**长度**（现按 **56.0 mm**）
- ③ 顺手量：板面总宽、灯条宽度、（可选）外侧到外侧
- 不一致就报给我，改常量即可（改完全链路测距按新尺寸走）
- 为什么重要：单目测距 = 已知尺寸 × 像素比例，尺寸差 1% → 距离差 1%。**合成自测测不出这类系统误差**（自己跟自己对），必须用实物对

**A2. 卷尺校距（需要相机上桌；10 分钟）** —— 验证"解算距离 vs 真实距离"：
- 摆法：相机固定不动；板**正对**镜头（尽量垂直于视线）；卷尺量**镜头前端面 → 板面**的直线距离
- 三档：**1.000 m / 2.000 m / 3.000 m**，每档停住 2 秒以上
- 推荐用新工具 `./build/pnp_view --serve=8081`（浏览器控制台）：看 `d=` 读数、青色重投影框是否压在绿色检测框上、`err=` 残差 px；**Save 按钮存标注快照**留证（落在 `/tmp/pnp_view/`，可拉回来我核对）
- 判据：1m 误差 <1cm；3m 误差 3~6cm（1~2%）属单目正常水平；**>5% 就先回去查 A1 的尺寸**；`err` 稳态应 ≲2px（仅限正对/小角度；大斜视 err 天然偏大）

**A3. 角度校 yaw + 确认 pitch 假设（可选，接着 A2 做，5 分钟）**：
- 板放 2m，转一个**已知角度**（量角器/桌边贴胶带做 20°/40° 标记）→ 看 `yaw=` 读数跟不跟
- 顺带定 pitch：板**立直**（垂直地面）用 `--armor_pitch=0`；按比赛实际 15° 倾角就用默认值——这个假设只影响 yaw、不影响距离

**A4. 静止连拍测噪声（1 分钟）**：板放 2m 静止不动，我录 ~200 帧统计关键点抖动 σ → 直接用作 tracker 的观测噪声初值

**A5. （相机上云台后）手眼标定（约 30 分钟，工具已就绪）**：棋盘格（11×8 方格 30mm，`configs/calibration.yaml` 已配好）固定在场地 → 云台带相机转 **≥15 个姿态**（板清晰入画、姿态多样）→ 采图 → 解算 → 覆盖 yaml：
```bash
./build/capture --serve=8080            # 浏览器：画面里 board OK 再存；Save/Auto/Quit 全在网页
./build/calibrate_robotworld_handeye > /tmp/robotworld.yaml   # 解算（约 1s/张）
# 把 stdout YAML 里的 R_camera2gimbal / t_camera2gimbal 两行覆盖进 configs/standard3.yaml
./build/pnp_view --serve=8081           # 应从 EXTR NOMINAL（红字）变成 EXTR CALIB（绿字）
```
**这之前世界系姿态是"相对相机的量"，距离/yaw 可用、绝对姿态不可信**（pnp_view 的 EXTR 提示 + solver 启动告警正是提醒这个）

**A6. （拿到 IMU/下位机姿态后）**：接云台姿态 → `Solver::set_R_gimbal2world()`（接口已留好），yaw 搜索中心会自动切到云台 yaw

## B. 我这边的开发顺序

1. **B1. tracker（EKF）← 主任务**：把逐帧位姿（噪声大、会跳）变成稳定状态（位置/速度 + yaw/角速度）。设计对齐同济 `tasks/auto_aim/tracker`；观测 = PnP 位姿；R 初值用 A4 实测 σ，Q/门限用合成轨迹调；里程碑 = 合成直线/圆弧轨迹收敛 + demo.avi 连续帧跟踪不丢
2. **B2. 真板数据汇总**：A1~A4 的数据回来后出"卷尺 vs 解算"对照表 → 决定要不要调尺寸常量或物点定义；顺势定性 f0060/f0600 的重投影残差地板（8~21px，疑关键点定义偏差）
3. **B3. 类别裁剪（只打步兵）**：丢非目标类别（G/前哨/基地），阈值重调
4. **B4. 瞄准（aimer）**：弹道解算 + 提前量（需要手眼外参真值 + 弹速；先写算法骨架，弹速下位机定了再填）
5. **B5. 传统法 180fps 原型**（台账开放项 B5）
6. **B6. 部署化**：开机脚本（nvpmodel + jetson_clocks）、engine 重建脚本、systemd 守护（台账 D12）

## C. 上实车那天的顺序

1. 板接 ttyACM0 → `./build/serial_test --mode=live --port=/dev/ttyACM0 --seconds=10 --send=1` 确认四元数链路通
2. 相机上云台 → **做 A5 手眼标定**（后面所有绝对姿态都依赖它；`capture --serve=8080` 边看边采）
3. `pnp_view --serve=8081` 对场地跑一遍（距离/yaw/重投影/外参状态一屏全有），Save 存标注快照 → 回去用真实数据验 B1/B4
4. 留意车上供电与散热（Jetson 降频会直接反映到 fps 上）

## D. 命令速查（现在就可用；参数必须 `=` 传值）

```bash
./build/detect_view --serve=8080                 # 纯检测（看框）
./build/pnp_view --serve=8081                    # 检测+解算+重投影+err+外参状态（A2/A3/场地检查用这个；Save 存快照）
./build/capture --serve=8080                     # 标定采集（画面 board OK 再存；下位机不在加 --no-q 只采图）
./build/calibrate_robotworld_handeye             # 手眼解算（读 assets/img_with_q；> 重定向留档）
./build/serial_test --mode=loop                  # 串口离线自检（15 项）
./build/serial_test --mode=live --send=1         # 串口实读（板接上后）
./build/solver_test --mode=synth                 # 解算自检：合成真值 σ=0 门 + 噪声/假设扫描
./build/solver_test --mode=image --save=out.png 图   # 真图：绿=检出点 红=重投影点（目视点序）
./build/solver_test --mode=bench                 # 解算耗时 p50/p99
./build/solver_test --mode=proj                  # 投影快路径等价性回归（改 solver 后跑）
```
