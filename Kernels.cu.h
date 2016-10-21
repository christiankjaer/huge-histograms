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
                               unsigned int num_blocks,       
                               int*         sgm_offset,
                               int*          block_sgm){

   const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

   /* block_sgm = [0, 0, 0, ...] */
   /* sgm_offset = [0, 37 , 1000, 201020, ...] */

   if (gid < num_blocks){
     //forall (int i = 0; i < num_blocks; i++) {
     int tmp = gid * block_size;
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

  __shared__ int Hsh[CHUNK_SIZE];

  // LARGEST TODO: I think gid should be multiplied or something by the stride 
  
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO: set if-statement to skip if gid (times thead_elems?) larger than tot_size..
  const unsigned int tid = blockIdx.x;
  const unsigned int bdx = blockDim.x;
  const unsigned int bid = tid * bdx; // Start of the current block.
  const unsigned int bnd = bid+bdx; // End of current block
  // TODO: consider verifying if below includes all elements in chunk (by ceil)..
  const unsigned int thread_elems = ceil( (float)CHUNK_SIZE / bdx);
  const unsigned int stride = ceil( (float)CHUNK_SIZE / thread_elems);
  
  // Get segment ID and offset
  unsigned int sgm_id = sgm_id_arr[bdx];
  unsigned int sgm_start = sgm_offset_arr[sgm_id];
  // Get last segment element idx
  unsigned int sgm_end;
  if (sgm_id != num_sgms-1)
    sgm_end = sgm_offset_arr[sgm_id+1]-1;
  else
    sgm_end = tot_size-1;
  
  // Check for possible conflict
  bool conflict = false;
  if (sgm_end < bnd) // TODO: verify this equality not off by one or something..
    conflict = true;

  // Essential kernel body:
  unsigned int elem;
  if (conflict) {             /* If conflicts, handle segment splits within block */
    while (smg_end < bnd) {
      // TODO ... handle segment split
      
      // Update start/end of segment
      sgm_id++;
      sgm_start = sgm_end;
      if (sgm_id != num_sgms-1)
	sgm_end = sgm_offset_arr[sgm_id+1]-1;
      else
	sgm_end = tot_size-1;
    }
  } else {                        /* If no conflict solve for trivial case */
    for (int i=0;i<num_threads;i++) {
      // Jump in strides
      elem = tid + i*stride; // current local hist element
      if ((elem+gid < tot_size) && (elem<CHUNK_SIZE))	
	atomicAdd(&Hsh[inds[elem]], 1); // Add to local shared histogram
      else
	break; // no need to do more strides
    }
  }
  
  __syncthreads();
  if (!conflict)
    for (int i=0;i<num_threads;i++) {
      elem = tid + i*stride; // current local hist element
      hist[elem + sgm_id*CHUNK_SIZE] = Hsh[elem]; // add to global hist
    }
  
}


#endif //KERNELS_HIST
