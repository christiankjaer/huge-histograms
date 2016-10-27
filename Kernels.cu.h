#ifndef KERNELS_HIST
#define KERNELS_HIST
#include "setup.cu.h"

#define GPU_HIST_SIZE 8192

// @summary : Computes the histogram indexes based on a normalization of the data
// @remarks : Assumes all values in the input array to be non-negative
// @params  : input_arr_d -> the input values
//          : hist_inds_d -> an array to write back the histogram indexes
//          : size_arr    -> the size of both arrays
//          : the largest input element
template <class T>
__global__ void histVals2IndexKernel(T*     input_arr_d,
                                     int*   hist_inds_d,
                                     int       size_arr,
                                     T        max_input){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size_arr){
    hist_inds_d[gid] = (int)(((float)input_arr_d[gid]/max_input)*(float)HISTOGRAM_SIZE);
  }
}

// @summary : computes the offsets
__global__ void segmentOffsets(int* inds_d,
                               int  inds_size,
                               int* segment_d, // sort of flag array.
                               int* segment_offsets_d,
                               int  num_segments){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < inds_size){
    int this_segment_max = CHUNK_SIZE;
    int this_segment     = 0;
    // TODO : find a smart formula for this.
    while (inds_d[gid] >= this_segment_max){
      this_segment_max += CHUNK_SIZE;
      this_segment++;
    }
    segment_d[gid] = this_segment;
    __syncthreads();
    if (gid == 0){
      segment_offsets_d[0] = 0;
    }
    else{
     if (this_segment != segment_d[gid-1]){
       segment_offsets_d[this_segment] = gid;
     }
    }
  }
}

// @remarks : needs proper documentation
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



/* Global segment counter, initilize before running segmentedHistKernel */
/* __device__ segmentCounter; */

/* All of the index calculations are probably wrong
 *
 * The sgmts array is an array of indices where the
 * segments start like
 * [0,501,2057,...]
 */

__global__ void segmentedHistKernel(unsigned int tot_size,
                                    unsigned int num_sgm,
                                    int *sgmts,
                                    int *inds,
                                    int *hist) {

  __shared__ int Hsh[CHUNK_SIZE];
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int bst = blockIdx.x * blockDim.x; // Start of the current block.
  const unsigned int bdx = blockDim.x;
  const unsigned int thread_elems = CHUNK_SIZE / bdx;
  // First and last element of segment inside block
  int start_segm;

  // Figure out which segment the current block starts in.

  // dummy segment counter
  int curr_segment = 0;
  if (threadIdx.x == 0) {
    while (sgmts[curr_segment] <= bst) {
      curr_segment++;
      //int current_val = atomicAdd(&segmentCounter, 1);
    }
    start_segm = sgmts[curr_segment];
  }
  __syncthreads();

  int end_segm = sgmts[curr_segment+1];
  /* While one of the gid's is in the current segment */
  while (true/* Somethings goes here */) {

    if (gid >= start_segm && gid < end_segm) {
      // write into shared histogram
      // using atomicAdd()
      atomicAdd(&Hsh[inds[gid] - start_segm], 1);
    }

    __syncthreads();
    // Copy the elements back to global memory
    for (int i = 0; i < thread_elems; i++) {
      hist[i * bdx + gid] = 1 ;/* Something here as well */
    }

    __syncthreads();

    /* if (sgmts[segmentCounter] - 1 == bst) { */
    /* } */
    curr_segment++;
    start_segm = sgmts[curr_segment];
    end_segm = sgmts[curr_segment+1];
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



// @summary: for each block, it finds respective segment which it belongs to

__global__ void blockSgmKernel(unsigned int block_size,
                               unsigned int num_chunks,
                               int*         sgm_offset,
                               int*          block_sgm){

  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  /* block_sgm = [0, 0, 0, 1, 2, 4, ...] */
  /* sgm_offset = [0, 37 , 1000, 201020, ...] */

  if (gid < num_chunks){
    //forall (int i = 0; i < num_blocks; i++) {
    int tmp = gid * block_size * CHUNK_SIZE;
    int j = 0;
    while (sgm_offset[j] >= tmp){
      j++;
    }
    block_sgm[gid] = j;
  }
}

__global__ void christiansHistKernel(unsigned int tot_size,
                                     unsigned int chunk_size,
                                     unsigned int num_sgms,
                                     unsigned int *sgm_idx,
                                     unsigned int *sgm_offset,
                                     unsigned int *inds,
                                     unsigned int *hist) {

  __shared__ int Hsh[GPU_HIST_SIZE];

  
  const unsigned int gid = blockIdx.x * blockDim.x * chunk_size + threadIdx.x;
  const unsigned int block_end = (blockIdx.x + 1) * blockDim.x * chunk_size;
  
  unsigned int curr_sgm = sgm_idx[blockIdx.x];

  unsigned int sgm_start = sgm_offset[curr_sgm];
  unsigned int sgm_end = (curr_sgm >= num_sgms - 1) ? sgm_offset[curr_sgm + 1] : tot_size;


  while (sgm_start < block_end) {
    // Reset the shared memory.
  
    for (unsigned int i = threadIdx.x; i < GPU_HIST_SIZE; i += blockDim.x) {
      Hsh[i] = 0;
    }

    __syncthreads();

    // The sequential loop.
    unsigned int offset = curr_sgm * GPU_HIST_SIZE;
    for (unsigned int i = gid; i < block_end; i += blockDim.x) {
      if (i < sgm_end) {
        atomicAdd(&Hsh[inds[i] - offset], 1);
      }
    }
    __syncthreads();
    
    // Write back to memory
    for (unsigned int i = threadIdx.x; i < GPU_HIST_SIZE; i += blockDim.x) {
      atomicAdd(&hist[offset + i], Hsh[i]);
    }

    curr_sgm++;
    sgm_start = sgm_offset[curr_sgm];
    sgm_end = (curr_sgm >= num_sgms - 1) ? sgm_offset[curr_sgm + 1] : tot_size;
  }

}


#endif //KERNELS_HIST
