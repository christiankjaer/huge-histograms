#ifndef KERNELS_HIST
#define KERNELS_HIST
#include <cuda_runtime.h>

#define GPU_HIST_SIZE 8192

template <class T>
__global__ void mapKer(unsigned int  tot_size,
                       unsigned int hist_size,
                       T             boundary,
                       T*                d_in,
                       int*             d_out){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < tot_size) {
    // ... implement f for index corresponding to indices.
    d_out[gid] = floor(d_in[gid]/boundary) * hist_size;
  }
}

__global__ void naiveHistKernel(unsigned int  tot_size,
                                int*              inds,
                                int*              hist) {

  __shared__ int Hsh[8192];

  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < tot_size) {
    atomicAdd(&hist[inds[gid]], 1);
  }

}

#endif KERNELS_HIST
