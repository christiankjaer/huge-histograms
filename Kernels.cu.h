#ifndef KERNELS_HIST
#define KERNELS_HIST
#include "setup.cu.h"

// @summary : Computes the histogram indexes based on a normalization of the data
// @remarks : Assumes all values in the input array to be non-negative
// @params  : input_arr_d -> the input values
//          : hist_inds_d -> an array to write back the histogram indexes
//          : size_arr    -> the size of both arrays
//          : the largest input element
template <class T>
__global__ void histVals2IndexKernel(T*            input_arr_d,
                                     unsigned int* hist_inds_d,
                                     unsigned int  size_arr,
                                     unsigned int  hist_size,
                                     T             max_input){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size_arr){
    hist_inds_d[gid] = (unsigned int)((input_arr_d[gid]/max_input)*(T)(hist_size-1));
  }
}

// @summary : computes the offsets
__global__ void segmentOffsetsKernel(unsigned int* inds_d,
                                     unsigned int  inds_size,
                                     unsigned int* segment_offsets_d){

  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < inds_size){
    // assumes inds already partially sorted
    int this_segment = inds_d[gid] / GPU_HIST_SIZE;
    __syncthreads();
    if (gid == 0){
      segment_offsets_d[0] = 0;
    } else {
      int prev_segment = inds_d[gid-1] / GPU_HIST_SIZE;
      if (this_segment != prev_segment){
        segment_offsets_d[this_segment] = gid;
      }
    }
  }
}

// @remarks : needs proper documentation
__global__ void naiveHistKernel(unsigned int  tot_size,
                                unsigned int* inds,
                                unsigned int* hist) {

  // Thread index
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < tot_size) {
    atomicAdd(&hist[inds[gid]], 1);
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


__global__ void histKernel(unsigned int tot_size,
                           unsigned int hist_size,
                           unsigned int chunk_size,
                           unsigned int *sgm_offset,
                           unsigned int *inds,
                           unsigned int *hist) {

  __shared__ int Hsh[GPU_HIST_SIZE];

  unsigned int gid = blockIdx.x * blockDim.x * chunk_size + threadIdx.x;
  const unsigned int block_end = (blockIdx.x + 1) * blockDim.x * chunk_size;

  unsigned int curr_sgm = inds[blockIdx.x*blockDim.x*chunk_size] / GPU_HIST_SIZE;
  unsigned int num_segments = ceil((float)hist_size / GPU_HIST_SIZE);

  unsigned int sgm_start = sgm_offset[curr_sgm];
  unsigned int sgm_end = (curr_sgm + 1 < num_segments) ? sgm_offset[curr_sgm + 1] : tot_size;

  sgm_start = sgm_offset[curr_sgm];
  sgm_end = (curr_sgm + 1 < num_segments) ? sgm_offset[curr_sgm + 1] : tot_size;


  while (sgm_start < block_end) {
    // Reset the shared memory.

    for (unsigned int i = threadIdx.x; i < GPU_HIST_SIZE; i += blockDim.x) {
      Hsh[i] = 0;
    }

    __syncthreads();

    // The sequential loop.
    unsigned int offset = curr_sgm * GPU_HIST_SIZE;
    for (; gid < min(block_end, sgm_end); gid += blockDim.x) {
      atomicAdd(&Hsh[inds[gid] - offset], 1);
    }
    __syncthreads();

    // Write back to memory
    for (unsigned int i = threadIdx.x; i < GPU_HIST_SIZE; i += blockDim.x) {
      if (offset + i < hist_size) {
         atomicAdd(&hist[i + offset], Hsh[i]);
      }
    }

    if (++curr_sgm == num_segments) break;

    sgm_start = sgm_offset[curr_sgm];
    sgm_end = (curr_sgm + 1 < num_segments) ? sgm_offset[curr_sgm + 1] : tot_size;
  }

}

#endif //KERNELS_HIST
