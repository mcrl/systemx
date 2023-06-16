// A simple program that computes the square root of a number
#include <cmath>
#include <iostream>
#include <string>

#include "systemxConfig.hpp"
#include "utils.hpp"

int main(int argc, char* argv[])
{

  std::cout << SYSTEMX_NAME << " " << SYSTEMX_VERSION_MAJOR << "." << SYSTEMX_VERSION_MINOR << std::endl;

  const double t = SYSTEMX::utils::gettime();
  std::cout << "Time " << t << std::endl;

  return 0;
}