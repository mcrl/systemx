// A simple program that computes the square root of a number
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <type_traits>

#include "spdlog/spdlog.h"
#include "json/json.h"
#include "cuda_runtime.h"

#include "systemxConfig.hpp"
#include "utils.hpp"
#include "driver.hpp"

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
  for (Json::ValueIterator iter = kernels.begin(); iter != kernels.end(); iter++) {
    auto op = (*iter)["op"].asString();
    auto gpus = asVector<uint>((*iter)["gpus"]);
    auto stream = (*iter)["stream"].asUInt();
    auto dimGrid = asVector<uint>((*iter)["dimGrid"]);
    auto dimBlock = asVector<uint>((*iter)["dimGrid"]);
    auto events = asVector<string>((*iter)["events"]);
    map<string, string> event_log_map;
    if ((*iter)["event_log_map"].isObject()) {
      for (Json::ValueIterator iter2 = (*iter)["event_log_map"].begin(); iter2 != (*iter)["event_log_map"].end(); iter2++) {
        event_log_map[iter2.key().asString()] = (*iter2).asString();
      }
    }

    for (uint gpu : gpus) {
      spdlog::info("Launch Kernel {0} on GPU {1}", op, gpu);
      
      // lazy init driver
      if (driver_map.find(gpu) == driver_map.end()) {
        driver_map[gpu] = new Driver(gpu);
      }
      // driver->launchKernel(kernel);
    }
  }
  // int gpu_index = 0;
  // vector<string> kernels;

  // Driver *driver;

  // // process command line args
  // if (checkCmdLineFlag(argc, (const char**)argv, "gpu")) {
  //   gpu_index = getCmdLineArgumentInt(argc, (const char **)argv, "gpu");
  // } else {
  //   gpu_index = 0;
  // }
  // if (gpu_index < ngpus) {
  //   driver = new Driver(gpu_index);
  //   spdlog::info("Using GPU {0}", gpu_index);
  // } else {
  //   throw runtime_error("GPU index out of range");
  // }

  // if (checkCmdLineFlag(argc, (const char **)argv, "kernels")) {
  //   char *_kernels;
  //   if (getCmdLineArgumentString(argc, (const char **)argv, "kernels", &_kernels)) {
  //     kernels = stringSplit(string(_kernels), ",");

  //     for (string kernel : kernels) {
  //       spdlog::info("Launching kernel {0}", kernel);
  //       // TODO: create thread to launch kernel
  //       driver->launchKernel(kernel);
  //     }
  //   } else {
  //     throw runtime_error("No kernels specified");
  //   }
  // } else {
  //   throw runtime_error("No kernels specified");
  // }

  // cleanup driver
  for (const auto& [id, driver] : driver_map) {
    delete driver;
  }

  return EXIT_SUCCESS;
}