cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(systemx 
  VERSION 1.0
  LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g -Wall -Wno-unused-result")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -gencode=arch=compute_70,code=sm_70")

file(GLOB CPP_SOURCES "*.cpp" "*.cc" "*.cxx")
file(GLOB CUDA_SOURCES "kernels/*.cu")

# Version control
configure_file(systemxConfig.hpp.in systemxConfig.hpp)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# Internal libraries
add_subdirectory(utils)

# External libraries

## CUDA
if(NOT DEFINED CUDA_INSTALL_PATH)
  set(CUDA_INSTALL_PATH "/usr/local/cuda")
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CUDA_ARCHITECTURES 70)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70)

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${PROJECT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/utils
  ${PROJECT_SOURCE_DIR}/kernels
  ${PROJECT_SOURCE_DIR}/external/spdlog/include
  ${CUDA_INSTALL_PATH}/include)

target_link_libraries(${PROJECT_NAME} PUBLIC 
  utils
  cudart)

target_link_directories(${PROJECT_NAME} PUBLIC 
  ${CUDA_INSTALL_PATH}/lib64)