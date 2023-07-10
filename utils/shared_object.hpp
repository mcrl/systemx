#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>

using namespace std;

namespace SYSTEMX {
namespace utils {

// Atomic counter that decrements to zero before signaling all threads in the 
// condition variable's waiting list.
class SharedCounter {
public:
  SharedCounter(uint initialValue) : decrementingCounter_(initialValue) {}
  void decrement() {
    mutex_.lock();
    decrementingCounter_--;

    unique_lock<mutex> lk(mutex_);
    cv_.wait(lk, [&] {return this->decrementingCounter_ == 0;}); // wait until counter is 0

    cv_.notify_one();
    mutex_.unlock();
  }

private:
  atomic<uint> decrementingCounter_;
  mutex mutex_;
  condition_variable cv_;
};
}
}