// A simple program that computes the square root of a number
#include <cmath>
#include <iostream>
#include <string>

#include "systemxConfig.h"

int main(int argc, char* argv[])
{
  if (argc < 2) {
    // TODO 12: Create a print statement using Tutorial_VERSION_MAJOR
    //          and Tutorial_VERSION_MINOR
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }

  std::cout << SYSTEMX_VERSION_MAJOR << "." << SYSTEMX_VERSION_MINOR << std::endl;

  // convert input to double
  const double inputValue = std::stod(argv[1]);

  // calculate square root
  const double outputValue = sqrt(inputValue);
  std::cout << "The square root of " << inputValue << " is " << outputValue
            << std::endl;
  return 0;
}