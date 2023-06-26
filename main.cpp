// A simple program that computes the square root of a number
#include <cmath>
#include <iostream>
#include <string>

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "systemxConfig.hpp"
#include "utils.hpp"
#include "driver.hpp"

using namespace SYSTEMX::utils;
using namespace SYSTEMX::core;
using namespace std;

int main(int argc, char *argv[])
{
  spdlog::info("{0}{1}.{2}", SYSTEMX_NAME, SYSTEMX_VERSION_MAJOR, SYSTEMX_VERSION_MINOR);

  int ngpus;
  CUDA_CALL(cudaGetDeviceCount(&ngpus));
  spdlog::info("Found {0} GPUs", ngpus);
  
  int gpu_index = 0;
  vector<string> kernels;

  Driver *driver;

  // process command line args
  if (checkCmdLineFlag(argc, (const char**)argv, "gpu")) {
    gpu_index = getCmdLineArgumentInt(argc, (const char **)argv, "gpu");
  } else {
    gpu_index = 0;
  }
  if (gpu_index < ngpus) {
    driver = new Driver(gpu_index);
    spdlog::info("Using GPU {0}", gpu_index);
  } else {
    throw runtime_error("GPU index out of range");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "kernels")) {
    char *_kernels;
    if (getCmdLineArgumentString(argc, (const char **)argv, "kernels", &_kernels)) {
      kernels = stringSplit(string(_kernels), ",");

      for (string kernel : kernels) {
        driver->launchKernel(kernel);
      }
    } else {
      throw runtime_error("No kernels specified");
    }
  } else {
    throw runtime_error("No kernels specified");
  }

  // cleanup
  delete driver;
  
  return EXIT_SUCCESS;
}