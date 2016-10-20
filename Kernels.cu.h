#ifndef KERNELS_HIST
#define KERNELS_HIST
#include "setup.cu.h"

template <class T>
__global__ void histIndKernel(float* input_arr_d,
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

  __shared__ int Hsh[CHUNCK_SIZE];


  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int bdx = blockDim.x;
  const unsigned int hist_elems = CHUNCK_SIZE / bdx;
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

__global__ void segmentedHistKernel(unsigned int tot_size,
                                    unsigned int num_sgm,
                                    int *sgmts,
                                    int *inds,
                                    int *hist) {

  __shared__ int Hsh[GPU_HIST_SIZE];

  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  while (/* */) {

    if (/* gid in working set */) {
      // write into shared histogram
      // using atomicAdd()
    }

    __syncthreads();
    // Write shared histogram to global histogram

    __syncthreads();
  }



}

// In the case of segments
// Keep index of current sub-histogram
// While (!all_segments_done)
//   if (gid in current segment)
//     do work
//   __syncthreads()
//   segment++
//   commit to memory

#endif //KERNELS_HIST
