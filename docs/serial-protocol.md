# 串口通信协议（视觉 ↔ 电控，'S','P' 帧）

> 状态：2026-10-11 落地。来源 = 同济 sp_vision_25 `io/gimbal`（电控按其实现，故照抄）。
> 实现：本仓库 `io/gimbal/gimbal.{hpp,cpp}`；自检工具 `tests/serial_test.cpp`（`--mode=loop` 逐项验证本文件所有字节级约定）。
> 链路：USB 虚拟串口（CDC ACM），默认 `/dev/ttyACM0`，115200（CDC 下波特率被忽略；若经 USB-TTL 桥接则必须一致）。

## 1. 帧格式（`__attribute__((packed))` 结构体直读写；**小端**）

### 1.1 下位机 → 视觉 `GimbalToVision`，共 **43 字节**

| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0 | u8[2] | head = `'S'`,`'P'` | 搜头同步（0x53 0x50） |
| 2 | u8 | mode | 0 空闲 / 1 自瞄 / 2 小符 / 3 大符 |
| 3 | f32×4 | q[4] | 云台 IMU 四元数，**wxyz 顺序** |
| 19 | f32 | yaw | rad |
| 23 | f32 | yaw_vel | rad/s |
| 27 | f32 | pitch | rad |
| 31 | f32 | pitch_vel | rad/s |
| 35 | f32 | bullet_speed | m/s |
| 39 | u16 | bullet_count | 子弹累计发送次数 |
| 41 | u16 | crc16 | **字节序 [低, 高]** |

### 1.2 视觉 → 下位机 `VisionToGimbal`，共 **29 字节**

| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0 | u8[2] | head = `'S'`,`'P'` | |
| 2 | u8 | mode | 0 不控制 / 1 控制云台不开火 / 2 控制且开火（实现：`control ? (fire ? 2 : 1) : 0`）|
| 3 | f32 | yaw | rad |
| 7 | f32 | yaw_vel | rad/s |
| 11 | f32 | yaw_acc | rad/s² |
| 15 | f32 | pitch | rad |
| 19 | f32 | pitch_vel | rad/s |
| 23 | f32 | pitch_acc | rad/s² |
| 27 | u16 | crc16 | **字节序 [低, 高]** |

## 2. CRC16 规格（与 `tools/crc.cpp` 表驱动实现逐位一致）

- 算法：**反射式 CRC-16/CCITT**（CRC-16/MCRF4XX 同款）
- 多项式：**0x1021**（反射写法 **0x8408**）
- 初值：**0xFFFF**；**无最终异或**；输入/输出均反射
- 计算范围：整帧去掉最后 2 字节 crc 字段
- 帧内字节序：**[低字节在前, 高字节在后]**（校验时 `(data[len-1] << 8) | data[len-2]`）
- 逐位写法（供电控对照）：

```c
uint16_t crc16 = 0xffff;
for (i = 0; i < len; i++) {
    crc16 ^= data[i];
    for (j = 0; j < 8; j++)
        crc16 = (crc16 & 1) ? (crc16 >> 1) ^ 0x8408 : (crc16 >> 1);
}
```

- 另有 CRC8（表驱动，init 0xff）一并移植备用（本协议未用）。

## 3. 收发时序约定（视觉侧实现语义）

- 收：读 2B 找 `'S','P'` → 读余下 41B → 整帧 CRC 校验 → 通过则入队（四元数 + 采样时刻）。
- 连续读错误 > 5000 次 → 关闭串口、1s 后重开（≤10 次），重开成功清空旧数据。
- 发：每帧填字段 → 算 CRC → 整帧写出；写失败仅告警不中断。
- 下位机侧建议以固定周期（≥100Hz）持续发送，视觉启动时会阻塞等待**第一帧**（可用配置 `first_q_timeout_s` 限定等待）。

## 4. 样例帧（由 `serial_test --mode=loop` 打印，供字节级对拍）

两帧均为人工填充的可辨识值（q=wxyz=(1,0.1,0.2,0.3)，yaw=0.5，pitch=-0.25，bullet_speed=25）：

```
GimbalToVision (43B): 53 50 01 00 00 80 3f cd cc cc 3d cd cc 4c 3e 9a 99 99 3e 00 00 00 3f 00 00 80 3f 00 00 80 be 00 00 c0 bf 00 00 c8 41 e8 03 92 d9
                      ↑head  ↑mode=1  ↑q.w=1.0      ↑q.x=0.1      ↑q.y=0.2      ↑q.z=0.3      ↑yaw=0.5     ↑yaw_vel=1.0  ↑pitch=-0.25 ↑pitch_vel=-1.5 ↑bullet_speed=25 ↑count=1000 ↑crc=0xd992（低 92 高 d9）

VisionToGimbal (29B): 53 50 02 00 00 00 3f 00 00 80 3f 00 00 00 40 00 00 80 be 00 00 c0 bf 00 00 00 c0 79 bb
                      ↑head  ↑mode=2  ↑yaw=0.5     ↑yaw_vel=1.0  ↑yaw_acc=2.0  ↑pitch=-0.25 ↑pitch_vel=-1.5 ↑pitch_acc=-2.0 ↑crc=0xbb79（低 79 高 bb）
```

> 43B 样例 crc16=0xd992、29B 样例 crc16=0xbb79，电控侧按 §2 算法算出的 crc 应与这两值一致（可当"表驱动实现是否正确"的第一道对拍）。

## 5. 自检与联调

```bash
./build/serial_test --mode=loop                                   # 离线：帧长/偏移/CRC/篡改全拒
./build/serial_test --mode=live --port=/dev/ttyACM0 --seconds=10 --send=1   # 实读下位机
```
