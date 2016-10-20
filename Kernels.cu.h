#ifndef KERNELS_HIST
#define KERNELS_HIST
#include "setup.cu.h"

// Maps array to histogram indices.
template <class T>
__global__ void histVals2IndexKernel(float* input_arr_d,
                                     int*   hist_inds_d,
                                     int    size_arr,
                                     float  max_input){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size_arr){
    hist_inds_d[gid] = (int)((input_arr_d[gid]/max_input)*(float)HISTOGRAM_SIZE);
  }
}

__global__ void naiveHistKernel(unsigned int  tot_size,
                                int*              inds,
                                int*              hist) {

  __shared__ int Hsh[CHUNK_SIZE];

  // Thread index
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // Block dimension (# of threads per block)
  const unsigned int bdx = blockDim.x;
  // # of threads assigned to each chunk
  const unsigned int hist_elems = CHUNK_SIZE / bdx;
  // # of tasks per thread
  const unsigned int tot_elems = tot_size / bdx;

  if (gid < tot_size) {
    for (int i = 0; i < hist_elems; i++) {
      Hsh[i*bdx + gid] = 0;
    }
    __syncthreads();
    for (int i = 0; i < tot_elems; i++) {
      atomicAdd(&Hsh[inds[bdx*i + gid]], 1);
    }
    __syncthreads();
    for (int i = 0; i < hist_elems; i++) {
      hist[i*bdx + gid] = Hsh[i*bdx + gid];
    }
  }

}

#endif //KERNELS_HIST
