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
__global__ void histVals2IndexKernel(T*     input_arr_d,
                                     int*   hist_inds_d,
                                     int       size_arr,
                                     T        max_input){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size_arr){
    hist_inds_d[gid] = (int)(((float)input_arr_d[gid]/max_input)*(float)HISTOGRAM_SIZE);
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

__global__ void hennesHistKernel(unsigned int tot_size,
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
  // Number of threads in the block
  const unsigned int bdx = blockDim.x;
  // Block index
  const unsigned int bid = blockIdx.x;
  // Global idx (first position of n strides)
  const unsigned int gidx = bid * CHUNK_SIZE * bdx + tidx;

  // (Global) Start of current block
  //const unsigned int bid = blockIdx.x * bdx;
  // (Global) End of current block
  const unsigned int bnd = gidx + CHUNK_SIZE * bdx;

  // TODO: consider verifying if below includes all elements in chunk (by ceil)..
  const unsigned int thread_elems = ceil( (float)CHUNK_SIZE / bdx);
  const unsigned int stride = bdx;

  // Get segment idx at start of block and its global offset
  unsigned int sgm_id = sgm_id_arr[bdx];
  unsigned int sgm_start = sgm_offset_arr[sgm_id];
  // Get last segment element global idx
  unsigned int sgm_end;
  if (sgm_id != num_sgms-1)
    sgm_end = sgm_offset_arr[sgm_id+1];
  else
    sgm_end = tot_size;

  // Check for possible conflict
  bool conflict = false;
  if (sgm_end < bnd) // TODO: verify this equality not off-by-1 or something..
    conflict = true;

  /* Zero out local histogram */
  for (int i = threadIdx.x; i < CHUNK_SIZE; i+=stride) {
    Hsh[i] = 0;
  }
  
  /***** If conflicts, handle segment splits within block *****/
  unsigned int local_elem;
  unsigned int global_elem;
  if (conflict) {
    while (sgm_end < bnd) {
      __syncthreads();
      // Jump in strides of block size (blockDim)
      // iterates by stride through the chunk
      for (int i=threadIdx.x;i<thread_elems;i+=stride) {
        global_elem = gidx + i; // global data point index
        // Update local histogram
        if ((global_elem < tot_size) && (i<CHUNK_SIZE)) {
          if ((global_elem>=sgm_start) && (global_elem<sgm_end))
            atomicAdd(&Hsh[inds_arr[i]], 1); // Add to local shared histogram
        }
      }

      __syncthreads();
      // TODO: Fix here
      // Update global histogram
      for (int i=threadIdx.x;i<thread_elems;i+=stride) {
        local_elem = i*stride;
        global_elem = gidx + i*stride;
        if ((global_elem < tot_size) && (local_elem<CHUNK_SIZE)) {
          if ((global_elem>=sgm_start) && (global_elem<sgm_end))
            break;
            //atomicAdd(&hist_arr[global_elem], Hsh[local_elem]);
        }
      }

      __syncthreads();
      /* Zero out local histogram */
      for (int i = threadIdx.x; i < CHUNK_SIZE; i+=stride) {
        Hsh[i] = 0;
      }

      // Update start/end of segment
      sgm_id++;
      sgm_start = sgm_end;
      if (sgm_id != num_sgms-1)
        sgm_end = sgm_offset_arr[sgm_id+1]-1;
      else
        sgm_end = tot_size-1;
    }
    /***** If no conflict solve for trivial case *****/
  } else {
    __syncthreads();
    // Jump in strides
    for (int i=threadIdx.x;i<CHUNK_SIZE;i+=stride) {
      // Current local hist element idx
      //local_elem = lidx + i*stride;
      global_elem = gidx + i;
      // Update local histogram
      if ((global_elem < tot_size) && (local_elem<CHUNK_SIZE))
        atomicAdd(&Hsh[global_elem], 1); // Add to local shared histogram
      else
        break; // no need to do more strides
    }
  }

  __syncthreads();
  // Only relevant for trivial case
  if (!conflict)
    // Update global histogram
    for (int i=threadIdx.x;i<CHUNK_SIZE;i+=stride) {
      //local_elem = lidx + i;
      global_elem = gidx + i; // global histogram index
      if ((global_elem < tot_size) && (i<CHUNK_SIZE)) {
        break;
        //atomicAdd(&hist[global_elem], Hsh[i]); // flushes local histogram to global histogram
      }
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
  unsigned int sgm_id = sgm_id_arr[bdx];
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
    for (int i=tidx;i<wload;i+=bdx) {
      global_elem = gidx + i; // global data point index
      if (global_elem < tot_size) {
        //Check if we are within a segment
        if ((global_elem>=sgm_start) && (global_elem<sgm_end)) 
          // global_index % CHUNK_SIZE = local_hist_id
          atomicAdd(&Hsh[inds_arr[global_elem]%CHUNK_SIZE], 1); // Atomic add for local histogram
      }
    }
    
    __syncthreads();
    // Update global histogram
    for (int i=threadIdx.x;i<CHUNK_SIZE;i+=bdx) {
      global_elem = sgm_id_arr[sgm_id]*CHUNK_SIZE + i; // Global histogram index
      // update histogram if we are within segment, and still inside global histogram size
      if (global_elem<sgm_id*CHUNK_SIZE && global_elem < sgm_end)
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
