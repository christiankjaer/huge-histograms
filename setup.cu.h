#ifndef SETUP

// @summary: Constant declarations for Main, Host and Kernels
#define GPU_HIST_SIZE        4096    // Chunk size for the GPU to work on, maximum 8192.
#define RADIX_END_BIT        8       // Number of bit for 1 byte
#define CUDA_BLOCK_SIZE      512     // Kernel blocksize
#define HARDWARE_PARALLELISM 65536   // 2^16

#endif //SETUP
