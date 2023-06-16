cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(systemx 
  VERSION 1.0
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB CPP_SOURCES "*.cpp" "*.cc" "*.cxx")
file(GLOB CUDA_SOURCES "kernels/*.cu")

# Version control
configure_file(systemxConfig.hpp.in systemxConfig.hpp)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# Internal libraries
add_subdirectory(utils)
target_link_libraries(${PROJECT_NAME} PUBLIC utils)

# External libraries
message(${PROJECT_BINARY_DIR})
# Library header files
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${PROJECT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/utils)