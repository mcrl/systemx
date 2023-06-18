// A simple program that computes the square root of a number
#include <cmath>
#include <iostream>
#include <string>

#include "spdlog/spdlog.h"

#include "systemxConfig.hpp"
#include "utils.hpp"

int main(int argc, char* argv[])
{
  spdlog::info("{0}{1}.{2} ({3})", SYSTEMX_NAME, SYSTEMX_VERSION_MAJOR, SYSTEMX_VERSION_MINOR, SYSTEMX::utils::gettime());
  
  return 0;
}