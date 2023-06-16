cmake_minimum_required(VERSION 3.10)

project(systemx 
  VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB CPP_SOURCES "*.cpp" "*.cc" "*.cxx")
file(GLOB CUDA_SOURCES "kernels/*.cu")

configure_file(systemxConfig.h.in systemxConfig.h)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR})