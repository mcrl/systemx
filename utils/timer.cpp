#include "timer.hpp"

#include <chrono>

using namespace std::chrono;

namespace SYSTEMX {
namespace utils {

double gettime() {
  using clock = steady_clock;
  using dns = duration<double, std::nano>;
  using tpns = time_point<clock, dns>;

  tpns base = clock::now();
  tpns curr = clock::now();
  return (double)((curr - base).count() / 1e9);
}
}
}