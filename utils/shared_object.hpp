#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <map>

using namespace std;

namespace SYSTEMX {
namespace utils {

// Atomic counter that decrements to zero before signaling all threads in the 
// condition variable's waiting list.
class SharedCounter {
public:
  SharedCounter(const uint &initialValue) : decrementingCounter_(initialValue) {}
  
  void decrement() {
    unique_lock<mutex> lk(mutex_); // locks mutex_ by calling mutex_.lock()
    decrementingCounter_--;
    cv_.wait(lk, [&] {return this->decrementingCounter_ == 0;}); // wait until counter is 0
    cv_.notify_one();
  }

private:
  atomic<uint> decrementingCounter_;
  mutex mutex_;
  condition_variable cv_;
};

typedef std::map<string, SharedCounter*> shared_counter_map_t;

}
}