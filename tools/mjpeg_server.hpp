#ifndef TOOLS__MJPEG_SERVER_HPP
#define TOOLS__MJPEG_SERVER_HPP

// 内置 MJPEG HTTP 服务（header-only，依赖 OpenCV）：把共享槽里的"最新一帧画面"编码后推给浏览器。
//
// 用法：主线程画好 BGR 帧后 streamer.publish(disp)，另起线程 serve_mjpeg(port, streamer, quit, tag)。
//
// 注意：JPEG 编码放在服务线程（实测编码一次 ~15ms，放主循环会把整条检测管线从 60fps 拖到 27fps），
// 服务线程按自己的节奏（约 30fps）取最新帧编码+推送，主循环只管画。
// 注意：accept 用 500ms 超时轮询 quit——裸 accept() 会永久阻塞，信号退出时 join 卡死、
// 进程杀不干净、还占着相机（实测踩过），务必保留这个超时。

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "logger.hpp"

namespace tools
{

/// 供 MJPEG 客户端取"最新一帧画面（BGR）"的共享槽（丢旧留新）
struct MjpegStreamer
{
  std::mutex m;
  std::condition_variable cv;
  cv::Mat frame;
  uint64_t seq = 0;

  void publish(const cv::Mat & f)  // 拷贝一份（调用方会复用自己的 Mat）
  {
    {
      std::lock_guard<std::mutex> lock(m);
      f.copyTo(frame);
      seq++;
    }
    cv.notify_all();
  }

  bool latest(cv::Mat & out, uint64_t & last, int timeout_ms)
  {
    std::unique_lock<std::mutex> lock(m);
    cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return seq != last; });
    if (seq == last) return false;
    frame.copyTo(out);
    last = seq;
    return true;
  }
};

inline bool write_all(int fd, const void * data, size_t n)
{
  const char * p = (const char *)data;
  while (n > 0) {
    ssize_t w = ::write(fd, p, n);
    if (w <= 0) return false;
    p += w;
    n -= (size_t)w;
  }
  return true;
}

/// 阻塞式服务循环（放独立线程里跑）；quit 置位后返回。一次只服务一个浏览器连接。
inline void serve_mjpeg(
  int port, MjpegStreamer & s, std::atomic<bool> & quit, const std::string & tag)
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
  if (::bind(lfd, (sockaddr *)&addr, sizeof addr) != 0 || ::listen(lfd, 2) != 0) {
    logger()->error("[{}] 绑定端口 {} 失败（被占用？）", tag, port);
    ::close(lfd);
    return;
  }

  ifaddrs * ifa = nullptr;
  if (::getifaddrs(&ifa) == 0) {
    for (auto * p = ifa; p != nullptr; p = p->ifa_next) {
      if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
      char buf[INET_ADDRSTRLEN] = {0};
      auto * sin = (sockaddr_in *)p->ifa_addr;
      ::inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof buf);
      if (std::strcmp(buf, "127.0.0.1") != 0)
        logger()->info("[{}] 浏览器打开: http://{}:{}/  （网卡 {}）", tag, buf, port, p->ifa_name);
    }
    ::freeifaddrs(ifa);
  }

  while (!quit) {
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
    logger()->info("[{}] 浏览器已接入", tag);
    const char * hdr =
      "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n"
      "Cache-Control: no-store\r\n\r\n";
    if (!write_all(cfd, hdr, std::strlen(hdr))) {
      ::close(cfd);
      continue;
    }

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
      if (!write_all(cfd, head, (size_t)n) || !write_all(cfd, jpeg.data(), jpeg.size()) ||
          !write_all(cfd, "\r\n", 2))
        break;
      // 编码+推送的剩余时间补齐到 33ms → 浏览器约 30fps
      auto used = std::chrono::steady_clock::now() - t0;
      auto left = std::chrono::milliseconds(33) - std::chrono::duration_cast<std::chrono::milliseconds>(used);
      if (left > std::chrono::milliseconds(0)) std::this_thread::sleep_for(left);
    }
    ::close(cfd);
    logger()->info("[{}] 浏览器断开", tag);
  }
  ::close(lfd);
}

}  // namespace tools

#endif  // TOOLS__MJPEG_SERVER_HPP
