#ifndef TOOLS__CONTROL_SERVER_HPP
#define TOOLS__CONTROL_SERVER_HPP

// 浏览器控制台服务（header-only，依赖 mjpeg_server.hpp）：在 MJPEG 看画面的基础上，
// 给网页加控制按钮（Save / Auto / Quit）与状态回读（/status JSON）——VS Code 关了也能全流程操作。
//
// 用法：主循环画好帧 → cs.streamer.publish(disp)，每 ~0.5s 填 cs.set_status(json)；
//       另起线程 serve_control(port, cs, quit, tag)。主循环侧：
//         cs.save_requests（比上次见到的大 → 有一次存图请求）
//         cs.auto_sec      （自动存图间隔秒，0=关；网页可改）
//         cs.quit_requested（网页按了 Quit；serve_control 同时会置 quit）
//
// 路由：/ → 控制页（HTML）；/stream → MJPEG 流；/save、/auto?sec=N、/quit、/status → JSON
//
// 注意（与 mjpeg_server 的差异）：网页开着流的同时还会每秒轮询 /status——单线程服务会
// 互卡死，所以这里每条连接一个线程（连接收发都有超时，断连/卡死的客户端能自行退出，
// 服务线程退出时统一 join，不泄漏）。
// 注意：accept 用 500ms 超时轮询 quit（裸 accept() 会永久阻塞，退出时 join 卡死、相机被占）。

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "logger.hpp"
#include "mjpeg_server.hpp"

namespace tools
{

/// 主循环 ↔ 网页之间的共享状态
struct ControlServer
{
  MjpegStreamer streamer;                   // 画面槽（主循环 publish，服务线程编码推流）
  std::atomic<uint64_t> save_requests{0};   // 网页 Save → ++；主循环按"比上次见到的大"消费
  std::atomic<bool> quit_requested{false};  // 网页 Quit → true（主循环看到后收尾）
  std::atomic<int> auto_sec{0};             // 自动存图间隔秒（0=关；网页 /auto?sec=N 设置）
  bool show_auto = true;                    // 网页是否显示 Auto 按钮（无"自动存图"语义的工具置 false）


  void set_status(const std::string & json)
  {
    std::lock_guard<std::mutex> lock(m_);
    status_ = json;
  }
  std::string status()
  {
    std::lock_guard<std::mutex> lock(m_);
    return status_;
  }

private:
  std::mutex m_;
  std::string status_ = "{}";
};

namespace control_detail
{

inline void send_all_close(
  int fd, const std::string & body, const char * ctype = "text/plain; charset=utf-8",
  int code = 200)
{
  char head[256];
  int n = std::snprintf(
    head, sizeof head,
    "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
    code, code == 200 ? "OK" : "Not Found", ctype, body.size());
  write_all(fd, head, (size_t)n);
  write_all(fd, body.data(), body.size());
}

/// 控制页（HTML 是浏览器渲染，用 UTF-8 没问题；画面内叠加才必须 ASCII）
inline std::string control_page(const std::string & title, bool show_auto = true)
{
  std::string html =
    std::string("<!doctype html><html><head><meta charset=\"utf-8\"><title>") + title +
    "</title><style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}"
    "button{font-size:15px;padding:8px 14px;margin:0 6px 8px 0;cursor:pointer}"
    "#st{white-space:pre-wrap;font-family:monospace;background:#000;padding:8px;"
    "border:1px solid #444;max-width:900px}</style></head><body><h2>" +
    title +
    "</h2><div><img src=\"/stream\" style=\"max-width:100%;border:1px solid #444\"></div>"
    "<p><button onclick=\"fetch('/save')\">Save (s)</button>";
  if (show_auto)
    html +=
      "<button onclick=\"fetch('/auto?sec=0')\">Auto OFF</button>"
      "<button onclick=\"fetch('/auto?sec=2')\">Auto 2s</button>"
      "<button onclick=\"fetch('/auto?sec=5')\">Auto 5s</button>";
  html +=
    "<button onclick=\"fetch('/quit')\">Quit (q)</button></p>"
    "<pre id=\"st\">connecting...</pre>"
    "<script>document.addEventListener('keydown',function(e){"
    "if(e.key=='s')fetch('/save');if(e.key=='q')fetch('/quit')});"
    "setInterval(function(){fetch('/status').then(function(r){return r.text()})"
    ".then(function(t){document.getElementById('st').textContent=t})"
    ".catch(function(e){document.getElementById('st').textContent='server gone'})},1000);"
    "</script></body></html>";
  return html;
}

inline int query_int(const std::string & query, const std::string & key, int fallback)
{
  auto pos = query.find(key + "=");
  if (pos == std::string::npos) return fallback;
  return std::atoi(query.c_str() + pos + key.size() + 1);
}

/// MJPEG 推流（编码在本线程做，约 30fps——见 mjpeg_server.hpp 文件头）
inline void handle_stream(int cfd, MjpegStreamer & s, std::atomic<bool> & quit)
{
  const char * hdr =
    "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n"
    "Cache-Control: no-store\r\n\r\n";
  if (!write_all(cfd, hdr, std::strlen(hdr))) return;

  cv::Mat frame;
  std::vector<uchar> jpeg;
  uint64_t last = 0;
  while (!quit) {
    auto t0 = std::chrono::steady_clock::now();
    if (!s.latest(frame, last, 200)) continue;
    cv::imencode(".jpg", frame, jpeg, {cv::IMWRITE_JPEG_QUALITY, 80});
    char head[160];
    int n = std::snprintf(
      head, sizeof head, "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
      jpeg.size());
    if (
      !write_all(cfd, head, (size_t)n) || !write_all(cfd, jpeg.data(), jpeg.size()) ||
      !write_all(cfd, "\r\n", 2))
      break;
    auto used = std::chrono::steady_clock::now() - t0;
    auto left
      = std::chrono::milliseconds(33) - std::chrono::duration_cast<std::chrono::milliseconds>(used);
    if (left > std::chrono::milliseconds(0)) std::this_thread::sleep_for(left);
  }
}

inline void handle_client(
  int cfd, ControlServer & cs, std::atomic<bool> & quit, const std::string & tag)
{
  timeval rto{0, 500 * 1000};  // 请求头读取超时：卡死的客户端不拖住线程
  timeval sto{1, 0};           // 发送超时：write_all 失败即退
  ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof rto);
  ::setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &sto, sizeof sto);

  // 读请求（最多几包，凑出请求行；凑不出就按 404 处理，简单可靠）
  std::string req;
  char buf[1024];
  for (int i = 0; i < 4; i++) {
    ssize_t n = ::recv(cfd, buf, sizeof buf, 0);
    if (n <= 0) break;
    req.append(buf, (size_t)n);
    if (req.find("\r\n") != std::string::npos) break;
  }
  std::string path = "/", query;
  {
    auto sp1 = req.find(' ');
    auto sp2 = sp1 == std::string::npos ? std::string::npos : req.find(' ', sp1 + 1);
    if (sp2 != std::string::npos) path = req.substr(sp1 + 1, sp2 - sp1 - 1);
  }
  auto qpos = path.find('?');
  if (qpos != std::string::npos) {
    query = path.substr(qpos + 1);
    path = path.substr(0, qpos);
  }

  if (path == "/" || path == "/index.html") {
    send_all_close(cfd, control_page(tag, cs.show_auto), "text/html; charset=utf-8");
  } else if (path == "/status") {
    send_all_close(cfd, cs.status(), "application/json");
  } else if (path == "/save") {
    cs.save_requests++;
    send_all_close(cfd, "{\"ok\":true}", "application/json");
  } else if (path == "/auto") {
    int sec = query_int(query, "sec", 0);
    if (sec < 0) sec = 0;
    cs.auto_sec = sec;
    send_all_close(
      cfd, "{\"ok\":true,\"auto_sec\":" + std::to_string(sec) + "}", "application/json");
  } else if (path == "/quit") {
    send_all_close(cfd, "{\"ok\":true,\"bye\":true}", "application/json");
    cs.quit_requested = true;
    quit = true;
  } else if (path == "/stream") {
    handle_stream(cfd, cs.streamer, quit);
  } else {
    send_all_close(cfd, "404", "text/plain", 404);
  }
  ::close(cfd);
}

}  // namespace control_detail

/// 阻塞式服务循环（放独立线程里跑）；quit 置位后返回（并 join 掉所有连接线程）
inline void serve_control(
  int port, ControlServer & cs, std::atomic<bool> & quit, const std::string & tag)
{
  int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (lfd < 0) {
    logger()->error("[{}] 建 socket 失败", tag);
    return;
  }
  int one = 1;
  ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)port);
  if (::bind(lfd, (sockaddr *)&addr, sizeof addr) != 0 || ::listen(lfd, 16) != 0) {
    logger()->error("[{}] 绑定端口 {} 失败（被占用？）", tag, port);
    ::close(lfd);
    return;
  }

  ifaddrs * ifa = nullptr;
  if (::getifaddrs(&ifa) == 0) {
    for (auto * p = ifa; p != nullptr; p = p->ifa_next) {
      if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
      char ip[INET_ADDRSTRLEN] = {0};
      ::inet_ntop(AF_INET, &((sockaddr_in *)p->ifa_addr)->sin_addr, ip, sizeof ip);
      if (std::strcmp(ip, "127.0.0.1") != 0)
        logger()->info("[{}] 浏览器控制台: http://{}:{}/  （网卡 {}）", tag, ip, port, p->ifa_name);
    }
    ::freeifaddrs(ifa);
  }

  // 每条连接一个线程（网页同时开着 /stream 且每秒轮询 /status）；结束后由本循环回收 join，
  // 避免长会话说上千条连接堆积
  struct Client
  {
    std::thread t;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<Client> clients;

  while (!quit) {
    for (auto it = clients.begin(); it != clients.end();) {
      if (*it->done) {
        it->t.join();
        it = clients.erase(it);
      } else {
        ++it;
      }
    }

    // accept 带 500ms 超时轮询 quit（见文件头注释）
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(lfd, &rfds);
    timeval tv{0, 500 * 1000};
    if (::select(lfd + 1, &rfds, nullptr, nullptr, &tv) <= 0) continue;
    int cfd = ::accept(lfd, nullptr, nullptr);
    if (cfd < 0) {
      if (quit) break;
      continue;
    }
    auto done = std::make_shared<std::atomic<bool>>(false);
    clients.push_back(
      Client{
        std::thread([cfd, &cs, &quit, &tag, done] {
          control_detail::handle_client(cfd, cs, quit, tag);
          *done = true;
        }),
        done});
  }

  ::close(lfd);
  // 连接线程收发都有超时（/stream 每 200ms 看一次 quit）→ 能自行退出，这里 join 不卡
  for (auto & c : clients) {
    if (c.t.joinable()) c.t.join();
  }
}

}  // namespace tools

#endif  // TOOLS__CONTROL_SERVER_HPP
