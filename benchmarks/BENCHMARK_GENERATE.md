# Benchmark generation guild

1. Manipulate values in `main.cpp` to the desired benchmark configuration
2. Build SystemX 
3. Execute the built benchmark binary in `build/bin`
``` bash
cd build
./bin/benchmarks /path/to/output/json
```

## Properties
- op: Benchmark name. Must be defined in [kernels.hpp](../kernels/kernels.hpp).
- gpus: List of gpus to use
- stream: Logical stream id. Logical id starts from 0 and can be different from physical id.
- streamPriority: Priority to assign to the stream
- dimGrid: Grid dimension of the kernel
- dimBlock: Block dimension of the kernel
- steps: Steps to repeat the kernel
- sharedCounters: (Optional) List of counters to synchronize the drivers
- sharedBuffers: (Optional) List of pointers to the device buffers (possible over multiple devices) shared between the drivers
- events: (Optional) List of cuda events to record
- interactions: (Optional) Map for interaction relationships, e.g., reads from / writes to