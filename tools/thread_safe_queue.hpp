#ifndef TOOLS__THREAD_SAFE_QUEUE_HPP
#define TOOLS__THREAD_SAFE_QUEUE_HPP

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace tools
{
/// 有界线程安全队列（语义对齐 sp_vision_25 的同名组件）
///
/// 满时行为：
///   PopWhenFull = true  —— 丢最旧、保最新（相机等"只关心最新值"的场景用这个）
///   PopWhenFull = false —— 丢新帧（保持队列里是较早的数据）
template <typename T, bool PopWhenFull = false>
class ThreadSafeQueue
{
public:
  explicit ThreadSafeQueue(size_t max_size) : max_size_(max_size) {}

  void push(const T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_) return;

    if (queue_.size() >= max_size_) {
      if (PopWhenFull)
        queue_.pop();
      else
        return;
    }

    queue_.push(value);
    not_empty_.notify_one();
  }

  /// 阻塞取一个元素；队列被 close() 且取空时返回（value 不动）
  void pop(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });
    if (queue_.empty()) return;

    value = queue_.front();
    queue_.pop();
  }

  T pop()
  {
    T value{};
    pop(value);
    return value;
  }

  bool empty() const
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /// 唤醒所有阻塞中的消费者（用于退出）
  void close()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_.notify_all();
  }

private:
  std::queue<T> queue_;
  const size_t max_size_;
  bool closed_ = false;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
};
}  // namespace tools

#endif  // TOOLS__THREAD_SAFE_QUEUE_HPP
