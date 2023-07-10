// A simple program that computes the square root of a number
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <type_traits>
#include <condition_variable>
#include <mutex>
#include <map>

#include "spdlog/spdlog.h"
#include "json/json.h"
#include "cuda_runtime.h"

#include "systemxConfig.hpp"
#include "utils.hpp"
#include "driver.hpp"
#include "kernels.hpp"

using namespace SYSTEMX::utils;
using namespace SYSTEMX::core;
using namespace std;

static void print_help(const char *prog_name) {
  printf("Usage: %s [--benchmark_file]\n", prog_name);
  printf("Options:\n");
  printf("     --benchmark_file : json benchmark file\n");
}

template<typename T>
vector<T> asVector(Json::Value &value) {
  vector<T> vec;
  T item;
  for (Json::ValueIterator iter = value.begin(); iter != value.end(); ++iter) {
    if constexpr (is_same<T, uint>::value)
      item = ((*iter).asUInt());
    else if constexpr (is_same<T, int>::value)
      item = ((*iter).asInt());
    else if constexpr (is_same<T, float>::value)
      item = ((*iter).asFloat());
    else if constexpr (is_same<T, double>::value)
      item = ((*iter).asDouble());
    else if constexpr (is_same<T, string>::value)
      item = ((*iter).asString());
    else if constexpr (is_same<T, bool>::value)
      item = ((*iter).asBool());
    else
      throw runtime_error("Unsupported type");
    vec.push_back(item);
  }
  return vec;
}

int main(int argc, char *argv[])
{
  spdlog::set_level(spdlog::level::trace); // TODO: set this to off for production
  spdlog::info("{0}{1}.{2}", SYSTEMX_NAME, SYSTEMX_VERSION_MAJOR, SYSTEMX_VERSION_MINOR);

  int ngpus;
  map<int, Driver *> driver_map;
  CUDA_CALL(cudaGetDeviceCount(&ngpus));
  spdlog::info("Found {0} GPUs", ngpus);

  // Parse benchmark file
  Json::Reader reader;
  Json::Value root;
  
  if (checkCmdLineFlag(argc, (const char **)argv, "benchmark_file")) {
    char *_benchmark_file;
    if (getCmdLineArgumentString(argc, (const char **)argv, "benchmark_file", &_benchmark_file)) {
      char *extension;
      getFileExtension(_benchmark_file, &extension);
      if (strcmp(extension, "json"))
        throw runtime_error("Invalid benchmark file specified");
      
      ifstream benchmark_file;
      benchmark_file.open(_benchmark_file);
      if (benchmark_file.is_open()) {
        reader.parse(benchmark_file, root);
        benchmark_file.close();
      } else {
        throw runtime_error("Unable to open benchmark file");
      }
    } else {
      throw runtime_error("Invalid benchmark file specified");
    }
  } else {
    print_help(argv[0]);
    exit(EXIT_FAILURE);
  }

  Json::Value kernels = root["kernels"];
  uint id = 0;
  for (Json::ValueIterator iter = kernels.begin(); iter != kernels.end(); iter++, id++) {
    auto _op = (*iter)["op"].asString();
    auto _gpus = asVector<uint>((*iter)["gpus"]);
    auto _stream = (*iter)["stream"].asUInt();
    auto _streamPriority = (*iter)["streamPriority"].asInt();
    auto _dimGrid = asVector<uint>((*iter)["dimGrid"]);
    auto _dimBlock = asVector<uint>((*iter)["dimBlock"]);
    auto _steps = (*iter)["steps"].asUInt();
    auto _events = asVector<string>((*iter)["events"]);

    /* optional shared arguments */
    // since shared arguments are pointers to heap variables, they **must** be dynamically
    // allocated and assigned as a member to `kargs`.
    map<string, SharedCounter*> *_shared_counter_map = new map<string, SharedCounter*>;
    if ((*iter)["sharedCounters"]) {
      for (const auto &counter : (*iter)["sharedCounters"]) {
        cout << counter.asString() << endl;
        (*_shared_counter_map)[counter.asString()] = new SharedCounter(_gpus.size());
      }
    }

    for (uint gpu : _gpus) {
      CUDA_CALL(cudaSetDevice(gpu));
      
      // lazy init driver
      if (driver_map.find(gpu) == driver_map.end()) {
        driver_map[gpu] = new Driver(gpu);
      } 
      
      // init kernel_run_args
      kernel_run_args *kargs = new kernel_run_args;
      kargs->id = id;
      kargs->stream = driver_map[gpu]->getStream(_stream, _streamPriority);
      kargs->gpus = _gpus;
      kargs->dimGrid = dim3(_dimGrid[0], _dimGrid[1], _dimGrid[2]);
      kargs->dimBlock = dim3(_dimBlock[0], _dimBlock[1], _dimBlock[2]);
      kargs->steps = _steps;
      
      // init events
      map<string, event_tuple_t> _event_map; // temporary data structure
      for (string _event : _events) {
        cudaEvent_t _;
        CUDA_CALL(cudaEventCreate(&_));
        event_tuple_t event(_event, _);
        _event_map[_event] = event;
        kargs->events.push_back(event);
      }

      // add optional shared arguments
      kargs->shared_counter_map = _shared_counter_map;
      
      driver_map[gpu]->launchKernel(_op, kargs);
    }
  }

  // TODO: cleanup kargs

  // cleanup driver
  for (const auto &[id, driver] : driver_map) {
    delete driver;
  }

  return EXIT_SUCCESS;
}