cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(systemx 
  VERSION 1.0
  LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -g -Wall -Wno-unused-result")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -gencode=arch=compute_70,code=sm_70")

file(GLOB CPP_SOURCES "*.cpp" "*.cc" "*.cxx")
file(GLOB CUDA_SOURCES "kernels/*.cu")

# Version control
configure_file(systemxConfig.hpp.in systemxConfig.hpp)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# External libraries

## spdlog
add_subdirectory(${PROJECT_SOURCE_DIR}/external/spdlog)
include_directories(${PROJECT_SOURCE_DIR}/external/spdlog/include)
list(APPEND PUBLIC_LIBS spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)

## jsoncpp
add_subdirectory(${PROJECT_SOURCE_DIR}/external/jsoncpp EXCLUDE_FROM_ALL)
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${JSON_INC_PATH})
link_libraries(jsoncpp_lib)

## CUDA
if(NOT DEFINED CUDA_INSTALL_PATH)
  set(CUDA_INSTALL_PATH "/usr/local/cuda")
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CUDA_ARCHITECTURES 70)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70)
list(APPEND PUBLIC_LIBS cudart cublas)
list(APPEND INCLUDE_DIRECTORIES ${CUDA_INSTALL_PATH}/include)
list(APPEND LINK_DIRECTORIES ${CUDA_INSTALL_PATH}/lib64)

# Internal libraries

## utils
add_subdirectory(${PROJECT_SOURCE_DIR}/utils)
target_include_directories(utils PRIVATE ${CUDA_INSTALL_PATH}/include)
target_include_directories(utils PRIVATE ${PROJECT_SOURCE_DIR}/external/spdlog/include)
list(APPEND INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/utils)
list(APPEND PUBLIC_LIBS utils)

## benchmarks
add_subdirectory(${PROJECT_SOURCE_DIR}/benchmarks)

# Target
list(APPEND INCLUDE_DIRECTORIES
  ${PROJECT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/kernels)

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORIES})
target_link_directories(${PROJECT_NAME} PUBLIC ${LINK_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${PUBLIC_LIBS})