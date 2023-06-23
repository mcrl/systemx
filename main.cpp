// A simple program that computes the square root of a number
#include <cmath>
#include <iostream>
#include <string>

#include "spdlog/spdlog.h"

#include "systemxConfig.hpp"
#include "utils.hpp"

using namespace SYSTEMX::utils;
using namespace std;

int main(int argc, char *argv[])
{
  spdlog::info("{0}{1}.{2}", SYSTEMX_NAME, SYSTEMX_VERSION_MAJOR, SYSTEMX_VERSION_MINOR);

  int gpu_index = 0;
  vector<string> kernels;
  
  // process command line args
  if (checkCmdLineFlag(argc, (const char**)argv, "gpu")) {
    gpu_index = getCmdLineArgumentInt(argc, (const char **)argv, "gpu");
  } else {
    gpu_index = 0;
  }
  // TODO: cudaGetDeviceCount < gpu_index, throw error
  spdlog::info("Using GPU {0}", gpu_index);

  if (checkCmdLineFlag(argc, (const char **)argv, "kernels")) {
    char *_kernels;
    if (getCmdLineArgumentString(argc, (const char **)argv, "kernels", &_kernels)) {
      kernels = stringSplit(string(_kernels), ",");

      for (string kernel : kernels) {
        spdlog::info("Launching kernel: {0}", kernel);
      }
    } else {
      throw runtime_error("No kernels specified");
    }
  } else {
    throw runtime_error("No kernels specified");
  }

  return EXIT_SUCCESS;
}