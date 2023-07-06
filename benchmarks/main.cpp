#include <stdlib.h>

#include <fstream>

#include "json/json.h"
#include "spdlog/spdlog.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <benchmark_file>\n", argv[0]);
    exit(1);
  }

  spdlog::info(argv[1]);

  Json::Value root;

	Json::Value kernels;

	Json::Value kernel_1;
  kernel_1["op"] = "idle";
  Json::Value kernel_1_gpus;
  kernel_1_gpus.append(0);
  kernel_1["gpus"] = kernel_1_gpus;
  kernel_1["stream"] = 0;
  kernel_1["streamPriority"] = 0;
  Json::Value kernel_1_dimGrid;
  kernel_1_dimGrid.append(80);
  kernel_1_dimGrid.append(1);
  kernel_1_dimGrid.append(1);
  kernel_1["dimGrid"] = kernel_1_dimGrid;
  Json::Value kernel_1_dimBlock;
  kernel_1_dimBlock.append(1024);
  kernel_1_dimBlock.append(1);
  kernel_1_dimBlock.append(1);
  kernel_1["dimBlock"] = kernel_1_dimBlock;
  kernel_1["steps"] = 1;
  Json::Value kernel_1_events;
  kernel_1_events.append("start");
  kernel_1_events.append("end");
  kernel_1["events"] = kernel_1_events;
  kernels.append(kernel_1);

	Json::Value kernel_2;
  kernel_2["op"] = "idle";
  Json::Value kernel_2_gpus;
  kernel_2_gpus.append(0);
  kernel_2["gpus"] = kernel_2_gpus;
  kernel_2["stream"] = 1;
  kernel_2["streamPriority"] = 0;
  Json::Value kernel_2_dimGrid;
  kernel_2_dimGrid.append(80);
  kernel_2_dimGrid.append(1);
  kernel_2_dimGrid.append(1);
  kernel_2["dimGrid"] = kernel_2_dimGrid;
  Json::Value kernel_2_dimBlock;
  kernel_2_dimBlock.append(1024);
  kernel_2_dimBlock.append(1);
  kernel_2_dimBlock.append(1);
  kernel_2["dimBlock"] = kernel_2_dimBlock;
  kernel_2["steps"] = 1;
  Json::Value kernel_2_events;
  kernel_2_events.append("start");
  kernel_2_events.append("end");
  kernel_2["events"] = kernel_2_events;
  kernels.append(kernel_2);
  
  root["kernels"] = kernels;
  
  Json::StyledWriter writer;
	auto str = writer.write(root);
  std::ofstream output_file(argv[1]);
  output_file << str;
  output_file.close();

  return EXIT_SUCCESS;
}