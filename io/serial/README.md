# serial（vendored，勿改）

wjwwood/serial 的副本（William Woodall & John Harrison，2012 版），逐字节取自同济
sp_vision_25 的 `io/serial/`（v1.1.2 树）。**MIT 许可，全文内嵌在各源文件头部**
（见 `include/serial/serial.h`、`src/serial.cc`、`src/impl/unix.cc` 等）。

为何 vendor 而不是自写 termios：与同济/电控联调链路逐字节对齐、零适配风险。
上游此版本作为普通 CMake 子目录（`add_subdirectory`）直接可用：Linux 走
`src/impl/unix.cc` + `src/impl/list_ports/list_ports_linux.cc`（POSIX termios，无 arch 分支）。

**注意（io/gimbal 依赖的行为）**：库默认 Timeout 全 0（termios VMIN=0/VTIME=0）→
`read()` 非阻塞立即返回。io/gimbal 的错误计数/重连逻辑依赖此语义（链路静默时
≈1Hz 重连循环），勿随手改默认超时。
