#ifndef SETUP

// @summary: Constant declarations for Main, Host and Kernels
#define GPU_HIST_SIZE        8192       // Chunk size for the GPU to work on, maximum 8192.
#define RADIX_END_BIT        8           // Number of bit for 1 byte
#define CUDA_BLOCK_SIZE      512         // Kernel blocksize
#define HARDWARE_PARALLELISM 65536       // 2^16
#define MAXIMUM_STREAM_SIZE  1024*1024   // Half the m
#define STREAM_SIZE_GPU      256*1024    // memory limit on GPU
#define CUDA_DEVICE_MEMORY   1024*1024*2 // memory on GPU

#endif //SETUP
