// 串口协议自检工具（离线 loop / 实读 live 双模）
//
// 注意：OpenCV 的参数解析器必须用 "=" 传值，写 --send 1 会把 1 当成位置参数！
// 用法示例：
//   ./serial_test --mode=loop                                     # 离线：帧长/偏移/CRC/单字节篡改/队列语义
//   ./serial_test --mode=live --wait=3                            # 实读：等首帧 3s（无板时验证错误路径）
//   ./serial_test --mode=live --port=/dev/ttyACM0 --seconds=10 --send=1
//
// live 返回码：0 正常 / 3 打开失败或首帧超时 / 4 设备不存在
// loop 全绿打印 "ALL PASS (n/n)"，任一失败返回 1

#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "cli_common.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/crc.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/yaml.hpp"

namespace
{
int g_pass = 0;
int g_fail = 0;

void check(bool ok, const std::string & name, const std::string & detail = "")
{
  (ok ? g_pass : g_fail)++;
  std::printf(
    "%-4s %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : "  --  ",
    detail.c_str());
}

std::string bytes_hex(const uint8_t * p, size_t n)
{
  std::string s;
  char buf[4];
  for (size_t i = 0; i < n; i++) {
    std::snprintf(buf, sizeof buf, "%02x", p[i]);
    s += buf;
    if (i + 1 < n) s += ' ';
  }
  return s;
}

int run_loop()
{
  using io::GimbalToVision;
  using io::VisionToGimbal;

  // ---- 1. 帧长与字段偏移（编译期锁死 + 运行期打印）----
  static_assert(sizeof(GimbalToVision) == 43, "GimbalToVision 必须是 43 字节");
  static_assert(sizeof(VisionToGimbal) == 29, "VisionToGimbal 必须是 29 字节");
  static_assert(
    offsetof(GimbalToVision, head) == 0 && offsetof(GimbalToVision, mode) == 2 &&
    offsetof(GimbalToVision, q) == 3 && offsetof(GimbalToVision, yaw) == 19 &&
    offsetof(GimbalToVision, bullet_count) == 39 && offsetof(GimbalToVision, crc16) == 41);
  static_assert(
    offsetof(VisionToGimbal, head) == 0 && offsetof(VisionToGimbal, mode) == 2 &&
    offsetof(VisionToGimbal, yaw) == 3 && offsetof(VisionToGimbal, pitch) == 15 &&
    offsetof(VisionToGimbal, crc16) == 27);
  check(true, "帧长/偏移编译期断言 (43B / 29B)");
  std::printf(
    "      GimbalToVision: size=%zu mode=%zu q=%zu yaw=%zu bullet_count=%zu crc16=%zu\n",
    sizeof(GimbalToVision), offsetof(GimbalToVision, mode), offsetof(GimbalToVision, q),
    offsetof(GimbalToVision, yaw), offsetof(GimbalToVision, bullet_count),
    offsetof(GimbalToVision, crc16));
  std::printf(
    "      VisionToGimbal: size=%zu mode=%zu yaw=%zu pitch=%zu crc16=%zu\n",
    sizeof(VisionToGimbal), offsetof(VisionToGimbal, mode), offsetof(VisionToGimbal, yaw),
    offsetof(VisionToGimbal, pitch), offsetof(VisionToGimbal, crc16));

  // ---- 2. CRC16 往返 + 帧内字节序 [低, 高] ----
  GimbalToVision rx{};
  rx.mode = 1;
  rx.q[0] = 1.0f;
  rx.q[1] = 0.1f;
  rx.q[2] = 0.2f;
  rx.q[3] = 0.3f;
  rx.yaw = 0.5f;
  rx.yaw_vel = 1.0f;
  rx.pitch = -0.25f;
  rx.pitch_vel = -1.5f;
  rx.bullet_speed = 25.0f;
  rx.bullet_count = 1000;
  uint8_t * rx_bytes = reinterpret_cast<uint8_t *>(&rx);
  rx.crc16 = tools::get_crc16(rx_bytes, sizeof(rx) - sizeof(rx.crc16));

  check(
    tools::check_crc16(rx_bytes, sizeof(rx)), "GimbalToVision CRC16 往返",
    "crc16=0x" + [&] {
      char b[8];
      std::snprintf(b, sizeof b, "%04x", rx.crc16);
      return std::string(b);
    }() + " 字节流 [低,高]=[" + bytes_hex(rx_bytes + 41, 1) + ", " + bytes_hex(rx_bytes + 42, 1) + "]");

  // ---- 3. 单字节篡改必须全被拒 ----
  {
    int rejected = 0;
    uint8_t buf[sizeof(rx)];
    for (size_t i = 0; i < sizeof(rx); i++) {
      std::memcpy(buf, &rx, sizeof(rx));
      buf[i] ^= 0x01;
      if (!tools::check_crc16(buf, sizeof(rx))) rejected++;
    }
    check(
      rejected == (int)sizeof(rx), "GimbalToVision 单字节篡改全拒",
      std::to_string(rejected) + "/" + std::to_string(sizeof(rx)));
  }

  VisionToGimbal tx{};
  tx.mode = 2;
  tx.yaw = 0.5f;
  tx.yaw_vel = 1.0f;
  tx.yaw_acc = 2.0f;
  tx.pitch = -0.25f;
  tx.pitch_vel = -1.5f;
  tx.pitch_acc = -2.0f;
  uint8_t * tx_bytes = reinterpret_cast<uint8_t *>(&tx);
  tx.crc16 = tools::get_crc16(tx_bytes, sizeof(tx) - sizeof(tx.crc16));

  check(tools::check_crc16(tx_bytes, sizeof(tx)), "VisionToGimbal CRC16 往返");
  {
    int rejected = 0;
    uint8_t buf[sizeof(tx)];
    for (size_t i = 0; i < sizeof(tx); i++) {
      std::memcpy(buf, &tx, sizeof(tx));
      buf[i] ^= 0x01;
      if (!tools::check_crc16(buf, sizeof(tx))) rejected++;
    }
    check(
      rejected == (int)sizeof(tx), "VisionToGimbal 单字节篡改全拒",
      std::to_string(rejected) + "/" + std::to_string(sizeof(tx)));
  }

  // ---- 4. 样例帧十六进制（协议文档 docs/serial-protocol.md 直接引用）----
  std::printf("      样例 GimbalToVision (43B): %s\n", bytes_hex(rx_bytes, sizeof(rx)).c_str());
  std::printf("      样例 VisionToGimbal (29B): %s\n", bytes_hex(tx_bytes, sizeof(tx)).c_str());

  // ---- 5. 队列语义（Gimbal 依赖面：pop / front / pop_for / clear / close）----
  tools::ThreadSafeQueue<int> q{8};
  check(q.empty(), "队列初始为空");

  int v = 0;
  auto t0 = std::chrono::steady_clock::now();
  bool got = q.pop_for(v, std::chrono::milliseconds(50));
  double waited_ms
    = tools::delta_time(std::chrono::steady_clock::now(), t0) * 1000.0;
  check(
    !got && waited_ms >= 40 && waited_ms < 500, "空队列 pop_for 超时返回 false",
    "等了 " + std::to_string((int)waited_ms) + "ms");

  q.push(1);
  q.push(2);
  check(q.front() == 1 && !q.empty(), "front 窥视（不消费，仍是队首）");
  check(q.pop() == 1, "pop 顺序 #1");
  check(q.pop() == 2, "pop 顺序 #2");
  check(q.empty(), "取空后 empty");

  q.push(3);
  q.push(4);
  q.clear();
  check(q.empty(), "clear 清空剩余");

  q.push(5);
  check(q.pop_for(v, std::chrono::milliseconds(100)) && v == 5, "pop_for 正常取到元素");

  q.close();
  t0 = std::chrono::steady_clock::now();
  got = q.pop_for(v, std::chrono::milliseconds(200));
  waited_ms = tools::delta_time(std::chrono::steady_clock::now(), t0) * 1000.0;
  check(
    !got && waited_ms < 150, "close 后 pop_for 立即返回 false",
    "等了 " + std::to_string((int)waited_ms) + "ms");
  check(q.pop() == 0, "close 后 pop() 返回 T{}（int 即 0）");

  // ---- 汇总 ----
  if (g_fail == 0) {
    std::printf("ALL PASS (%d/%d)\n", g_pass, g_pass + g_fail);
    return 0;
  }
  std::printf("FAILED %d/%d\n", g_fail, g_pass + g_fail);
  return 1;
}

int run_live(cv::CommandLineParser & cli, const std::string & config_path)
{
  auto yaml = tools::load(config_path);

  std::string port = tools::read<std::string>(yaml, "com_port");
  cv::String port_override = cli.get<cv::String>("port");
  if (!port_override.empty()) port = port_override;
  int baud = cli.get<int>("baud");
  double wait_s = cli.get<double>("wait");
  double seconds = cli.get<double>("seconds");
  bool do_send = cli.get<int>("send") != 0;
  bool do_q = cli.get<int>("q") != 0;

  // 设备存在性预检：返回码更明确，不让构造里的异常吞掉原因
  if (::access(port.c_str(), F_OK) != 0) {
    tools::logger()->error("[serial_test] 设备不存在: {}（下位机没插？ttyACM 编号插拔后会变）", port);
    return 4;
  }

  // 参数覆盖走临时 yaml（io::Gimbal 只认 config_path；跑完即删，不落仓库）
  yaml["com_port"] = port;
  if (baud > 0) yaml["baud_rate"] = baud;
  yaml["first_q_timeout_s"] = wait_s;
  std::string tmp_path = "/tmp/serial_test_" + std::to_string(::getpid()) + ".yaml";
  {
    YAML::Emitter out;
    out << yaml;
    std::ofstream fout(tmp_path);
    fout << out.c_str();
  }

  tools::logger()->info("[serial_test] 打开 {}（等首帧 {:.1f}s）...", port, wait_s);
  std::unique_ptr<io::Gimbal> gimbal;
  try {
    gimbal = std::make_unique<io::Gimbal>(tmp_path);
  } catch (const std::exception & e) {
    std::remove(tmp_path.c_str());
    tools::logger()->error("[serial_test] 打开失败/首帧超时: {}", e.what());
    return 3;
  }
  std::remove(tmp_path.c_str());

  tools::Exiter exiter;
  auto t_start = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    auto state = gimbal->state();
    auto mode = gimbal->mode();

    if (do_q) {
      auto qt = gimbal->q(std::chrono::steady_clock::now());
      tools::logger()->info(
        "[serial_test] q(w={:.4f} x={:.4f} y={:.4f} z={:.4f})", qt.w(), qt.x(), qt.y(), qt.z());
    }

    tools::logger()->info(
      "[serial_test] mode={} yaw={:.3f} yaw_vel={:.3f} pitch={:.3f} pitch_vel={:.3f} "
      "bullet_speed={:.1f} bullet_count={}",
      gimbal->str(mode), state.yaw, state.yaw_vel, state.pitch, state.pitch_vel,
      state.bullet_speed, state.bullet_count);

    if (do_send) gimbal->send(true, false, 0, 0, 0, 0, 0, 0);

    if (tools::delta_time(std::chrono::steady_clock::now(), t_start) >= seconds) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  gimbal->send(false, false, 0, 0, 0, 0, 0, 0);  // 收尾：通知下位机回到不控制
  tools::logger()->info("[serial_test] live 测试结束");
  return 0;
}

}  // namespace

const cv::String keys =
  "{help h usage ? |      | 打印帮助 }"
  "{@config-path  | configs/standard3.yaml | 配置文件路径（live 用 com_port 等键；loop 不用）}"
  "{mode m         | loop | loop = 离线自检（不碰串口）/ live = 实读下位机帧 }"
  "{port           |      | 覆盖 com_port（live 用，如 /dev/ttyACM0）}"
  "{baud           | 0    | 覆盖 baud_rate（>0 才生效）}"
  "{seconds s      | 5    | live 时长（秒）}"
  "{wait w         | 3    | live 等首帧超时（秒）；无板时用来快速验证错误路径 }"
  "{send           | 0    | 1 = live 期间周期发送控制帧（true,false,0...）}"
  "{q              | 0    | 1 = live 额外调用 q(t)（链路断会阻塞，默认不调）}";

int main(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<cv::String>("@config-path");
  if (!cli_tools::config_path_ok("serial_test", config_path)) return 1;

  const std::string mode = cli.get<cv::String>("mode");
  if (mode == "loop") return run_loop();
  if (mode == "live") return run_live(cli, config_path);

  tools::logger()->error("[serial_test] 未知模式: {}", mode);
  return 1;
}
