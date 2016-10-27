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
__global__ void histVals2IndexKernel(T*            input_arr_d,
                                     unsigned int* hist_inds_d,
                                     int              size_arr,
                                     T               max_input){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size_arr){
    hist_inds_d[gid] = (int)((input_arr_d[gid]/max_input)*(float)HISTOGRAM_SIZE);
  }
}

// @summary : computes the offsets
__global__ void segmentOffsets(unsigned int* inds_d,
                               unsigned int  inds_size,
                               unsigned int* segment_d, // sort of flag array.
                               unsigned int* segment_offsets_d,
                               unsigned int  block_workload,
                               unsigned int  block_size,
                               unsigned int* block_sgm_index_d){

  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < inds_size){
    // assumes inds already partially sorted
    int this_segment     = inds_d[gid] / CHUNK_SIZE;
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
    if ((gid % block_workload) == 0){
      block_sgm_index_d[gid/block_workload] = this_segment;
      //printf("%d - %d\n", gid, this_segment);
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

// @summary : computes the offsets
__global__ void segmentOffsets(int* inds_d,
                               int  inds_size,
                               int* segment_d, // sort of flag array.
                               int* segment_offsets_d){
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


__global__ void christiansHistKernel(unsigned int tot_size,
                                     unsigned int hist_size,
                                     unsigned int chunk_size,
                                     unsigned int *sgm_idx,
                                     unsigned int *sgm_offset,
                                     unsigned int *inds,
                                     unsigned int *hist) {

  __shared__ int Hsh[GPU_HIST_SIZE];

  const unsigned int gid = blockIdx.x * blockDim.x * chunk_size + threadIdx.x;
  const unsigned int block_end = (blockIdx.x + 1) * blockDim.x * chunk_size;

  unsigned int curr_sgm = sgm_idx[blockIdx.x];
  unsigned int num_segments = ceil((float)hist_size / GPU_HIST_SIZE);

  unsigned int sgm_start = sgm_offset[curr_sgm];
  unsigned int sgm_end = (curr_sgm + 1 < num_segments) ? sgm_offset[curr_sgm + 1] : tot_size;

  while (sgm_start < block_end) {
    // Reset the shared memory.

    for (unsigned int i = threadIdx.x; i < GPU_HIST_SIZE; i += blockDim.x) {
      Hsh[i] = 0;
    }

    __syncthreads();

    // The sequential loop.
    unsigned int offset = curr_sgm * GPU_HIST_SIZE;
    for (unsigned int i = gid; i < min(block_end, sgm_end); i += blockDim.x) {
      atomicAdd(&Hsh[inds[i] - offset], 1);
    }
    __syncthreads();

    // Write back to memory
    for (unsigned int i = threadIdx.x; i < GPU_HIST_SIZE; i += blockDim.x) {
      if (offset + i < hist_size) {
        atomicAdd(&hist[offset + i], Hsh[i]);
      }
    }

    if (++curr_sgm == num_segments) break;

    sgm_start = sgm_offset[curr_sgm];
    sgm_end = (curr_sgm + 1 < num_segments) ? sgm_offset[curr_sgm + 1] : tot_size;
  }

}

// @summary : Computes local histogram of CHUNK_SIZE, and commits to global histogram
// @remarks : Assumes all index arrays are non-negative values
// @params  : tot_size        -> Number of data entries
//          : num_chunks      -> Number of chunks in global histogram
//          : hist_arr        -> Global histogram
//          : inds_arr        -> Data entries to be accumulated in histogram
//          : num_sgms        -> Number of segments to be handle
//          : sgm_id_arr      -> Holds indexes for where each segment a block starts in
//          : sgm_offset_arr  -> 
//          : the largest input element
__global__ void grymersHistKernel(unsigned int tot_size,
                                  unsigned int num_chunks,
                                  unsigned int num_sgms,
                                  unsigned int *sgm_id_arr,
                                  unsigned int *sgm_offset_arr,
                                  unsigned int *inds_arr,
                                  unsigned int *hist_arr) {
  
  // Block local histogram
  __shared__ int Hsh[CHUNK_SIZE];

  // Local idx  (----//----)
  const unsigned int tidx = threadIdx.x;
  // Number of threads in the block/stride size
  const unsigned int bdx = blockDim.x;
  // Block index
  const unsigned int bid = blockIdx.x;
  // Block workload
  const unsigned int wload = CHUNK_SIZE*bdx;
  // Global idx (first position of n datapoints in global data)
  const unsigned int gidx = bid * wload + tidx;
  // (Global) End of current block to be worked on
  const unsigned int bnd = gidx + wload;

  // Get segment idx at start of block and its global offset
  unsigned int sgm_id    = sgm_id_arr[bdx];
  unsigned int sgm_start = sgm_offset_arr[sgm_id];
  // Get last segment element global idx
  unsigned int sgm_end;
  if (sgm_id != num_sgms-1)
    sgm_end = sgm_offset_arr[sgm_id+1];
  else
    sgm_end = tot_size;

  /* Zero out local histogram */
  for (int i = threadIdx.x; i < CHUNK_SIZE; i+=bdx) {
    Hsh[i] = 0;
  }
  
  unsigned int global_elem; // variable to decide which data entry, or global histogram index

  /* Loops through CHUNK_SIZE * blockDim.x data points to be added to histogram */
  /* forall i < CHUNK_SIZE * blockDim.x  */
    
  while (sgm_end < bnd) {
    __syncthreads();
    // Jump in strides of block size (blockDim)
    // iterates by stride through the CHUNK_SIZE*blockDim.x elements
    for (int i=gidx;i<gidx+wload;i+=bdx) {
      //global_elem = gidx + i; // global data point index
      if (i < tot_size) {
        //Check if we are within a segment
        if ((i>=sgm_start) && (i<sgm_end)) 
          // global_index % CHUNK_SIZE = local_hist_id
          atomicAdd(&Hsh[inds_arr[i]%CHUNK_SIZE], 1); // Atomic add for local histogram
      }
    }
    
    __syncthreads();
    // Update global histogram
    for (int i=tidx;i<CHUNK_SIZE;i+=bdx) {
      global_elem = sgm_id*CHUNK_SIZE + i; // Global histogram index
      // update histogram if we are within segment, and still inside global histogram size
      if (global_elem>=sgm_id*CHUNK_SIZE && global_elem < sgm_end)
        atomicAdd(&hist_arr[global_elem], Hsh[i]); // Atomic update for global histogram
    }
    
    __syncthreads();
    /* Zero out local histogram */
    for (int i = threadIdx.x; i < CHUNK_SIZE; i+=bdx) {
      Hsh[i] = 0;
    }
    
    // Update start/end of segment
    sgm_id++;
    sgm_start = sgm_end;
    if (sgm_id != num_sgms-1)
      sgm_end = sgm_offset_arr[sgm_id+1];
    else
      sgm_end = tot_size;
  }
}



#endif //KERNELS_HIST
