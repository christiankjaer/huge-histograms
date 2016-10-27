#ifndef HOST_HIST
#define HOST_HIST

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cub/cub.cuh>
#include "Kernels.cu.h"
#include "setup.cu.h"

// @summary : d_out = scan_exc (+) 0 d_in.
template<class T>
void prefixSumExc(unsigned long arr_size,
                  T*            d_in,
                  T*            d_out) {

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                d_in,
                                d_out,
                                arr_size);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                d_in,
                                d_out,
                                arr_size);
}

// @summary : computes the maximum value in an array of values of type T.
template<class T>
T maximumElement(T* d_in, int arr_size){

  // Determine temporary device storage requirements
  T* d_max;
  T  h_max;
  cudaMalloc((void**)&d_max, sizeof(T));
  void*    d_temp_storage     = NULL;
  size_t   temp_storage_bytes = 0;

  // Dummy call, to set temp_storage values.
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, arr_size);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Let Cub handle the reduction
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, arr_size);

  // Copy back the reduced element
  cudaMemcpy(&h_max, d_max, sizeof(T), cudaMemcpyDeviceToHost);

  // Clean up memory
  cudaFree(d_temp_storage);
  cudaFree(d_max);
  return h_max;
}


// @summary : Normalizes the dateset, and computes histogram indexes.
// @remarks : Asserts the value and index arrays to have the same sizes.
// @params  : block_size -> size of the CUDA blocks to be used
//            arr_size   -> size of the two arrays
//            boundary   -> the maximum element size
//            vals_h     -> a pointer to the host allocated values
//            inds_h     -> a pointer to the host allocated index array
template<class T>
void histVals2IndexDevice (unsigned int    arr_size,
                           T*                vals_d,
                           int*              inds_d){

  // Allocate device memory
  /* T*   vals_d; */
  /* int* inds_d; */
  /* cudaMalloc((void**)&vals_d, arr_size * sizeof(float)); */
  /* cudaMalloc((void**)&inds_d, arr_size * sizeof(int)); */
  /* cudaMemcpy(vals_d, vals_h, arr_size*sizeof(T), cudaMemcpyHostToDevice); */

  //  Figure out the boundaries (vague).
  int num_blocks = ceil(arr_size / CUDA_BLOCK_SIZE);
  T   boundary   = maximumElement<T>(vals_d, arr_size);

  // TODO : handle {arr_size > sizeof(shared_memory)} !?

  // Run indexing kernel
  histVals2IndexKernel<T><<<num_blocks, CUDA_BLOCK_SIZE>>>
    (vals_d, inds_d, arr_size, boundary);
  cudaThreadSynchronize();

  // Write back result
  //cudaMemcpy(inds_h, inds_d, arr_size * sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up memory
  //cudaFree(vals_d);
  //cudaFree(inds_d);
}


// @summary : Normalizes the dateset, and computes histogram indexes.
// @remarks : Asserts the value and index arrays to have the same sizes.
// @params  : block_size -> size of the CUDA blocks to be used
//            arr_size   -> size of the two arrays
//            boundary   -> the maximum element size
//            vals_h     -> a pointer to the host allocated values
//            inds_h     -> a pointer to the host allocated index array
template<class T>
void histVals2Index (unsigned int    arr_size,
                     T*                vals_h,
                     int*              inds_h){

  // Allocate device memory
  T*   vals_d;
  int* inds_d;
  cudaMalloc((void**)&vals_d, arr_size * sizeof(float));
  cudaMalloc((void**)&inds_d, arr_size * sizeof(int));
  cudaMemcpy(vals_d, vals_h, arr_size*sizeof(T), cudaMemcpyHostToDevice);

  //  Figure out the boundaries (vague).
  int num_blocks = ceil(arr_size / CUDA_BLOCK_SIZE);
  T   boundary   = maximumElement<T>(vals_d, arr_size);

  // TODO : handle {arr_size > sizeof(shared_memory)} !?

  // Run indexing kernel
  histVals2IndexKernel<T><<<num_blocks, CUDA_BLOCK_SIZE>>>
    (vals_d, inds_d, arr_size, boundary);
  cudaThreadSynchronize();

  // Write back result
  cudaMemcpy(inds_h, inds_d, arr_size * sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up memory
  cudaFree(vals_d);
  cudaFree(inds_d);
}

// @summary : partially sortes an index array in chunk size bounded segments
void radixSortDevice(int* unsorted_d,
                     int* sorted_d,
                     int  array_length){

  // Allocate device memory
  /* int* d_keys_in; */
  /* int* d_keys_out; */
  /* cudaMalloc(&d_keys_in, sizeof(int)*array_length); */
  /* cudaMalloc(&d_keys_out, sizeof(int)*array_length); */
  /* cudaMemcpy(d_keys_in, array_to_be_sorted, sizeof(int)*array_length, */
  /*            cudaMemcpyHostToDevice); */

  void   *d_temp_storage    = NULL;
  size_t temp_storage_bytes = 0;


  // TODO : Figure out an intelligent way of computing end bit [-.-]
  int begin_bit = ceil(log2((float) CHUNK_SIZE));
  // int end_bit   = max(begin_bit, (int)ceil(log2((float) HISTOGRAM_SIZE))) + 1;
  int end_bit   = sizeof(int)*8;

  // Figure out how much temporary storage is needed
  cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 unsorted_d,
                                 sorted_d,
                                 array_length,
                                 begin_bit,
                                 end_bit);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Call the library function
  cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 unsorted_d,
                                 sorted_d,
                                 array_length,
                                 begin_bit,
                                 end_bit);

  // Write back the result
  /* cudaMemcpy(array_to_be_sorted, d_keys_out, sizeof(int)*array_length, */
  /*            cudaMemcpyDeviceToHost); */

  // Clean up memory
  /* cudaFree(d_keys_in); */
  /* cudaFree(d_keys_out); */
  cudaFree(d_temp_storage);
}

// @summary : partially sortes an index array in chunk size bounded segments
void radixSort(int* array_to_be_sorted,
               int  array_length){

  // Allocate device memory
  int* d_keys_in;
  int* d_keys_out;
  cudaMalloc(&d_keys_in, sizeof(int)*array_length);
  cudaMalloc(&d_keys_out, sizeof(int)*array_length);
  cudaMemcpy(d_keys_in, array_to_be_sorted, sizeof(int)*array_length,
             cudaMemcpyHostToDevice);

  void   *d_temp_storage    = NULL;
  size_t temp_storage_bytes = 0;


  // TODO : Figure out an intelligent way of computing end bit [-.-]
  int begin_bit = ceil(log2((float) CHUNK_SIZE));
  // int end_bit   = max(begin_bit, (int)ceil(log2((float) HISTOGRAM_SIZE))) + 1;
  int end_bit   = sizeof(int)*8;

  // Figure out how much tempoary storage is needed
  cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_keys_out,
                                 array_length,
                                 begin_bit,
                                 end_bit);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Call the library function
  cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_keys_out,
                                 array_length,
                                 begin_bit,
                                 end_bit);

  // Write back the result
  cudaMemcpy(array_to_be_sorted, d_keys_out, sizeof(int)*array_length,
             cudaMemcpyDeviceToHost);

  // Clean up memory
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
}

// @summary: Constructs a histogram
template <class T>
void naiveHist(T*      h_array,
               int*     h_hist,
               int data_length) {

  // histogram index array
  int* d_inds;
  cudaMalloc(&d_inds, sizeof(int) * data_length);

  // Finds maximum element
  // int max_elem = 10000;

}

// @summary : for now, just computes segment_sizes.
void metaData(unsigned int  inds_size,
              unsigned int* inds_d,
              unsigned int  num_segments,
              unsigned int* segment_sizes_d
              ){
  int num_blocks = ceil(inds_size / CUDA_BLOCK_SIZE);
  unsigned int* segment_d;
  cudaMalloc(&segment_d, sizeof(unsigned int)*inds_size);
  cudaMemset(segment_d, 0, inds_size * sizeof(unsigned int));
  cudaMemset(segment_sizes_d, 0, num_segments * sizeof(unsigned int));
  segmentOffsets<<<num_blocks, CUDA_BLOCK_SIZE>>>
    (inds_d, inds_size, segment_d, segment_sizes_d);
  cudaThreadSynchronize();
}

// @summary: finds index for segment offset, for each block
void blockSgm (unsigned int  block_size,
               unsigned int  tot_size,
               unsigned int* sgm_offset,
               unsigned int* block_sgm){
  // number of chunks to be worked on
  const unsigned int num_chunks = ceil((float) tot_size / (CHUNK_SIZE*block_size));
  // Number of blocks to construct block segment array
  const unsigned int num_blocks = ceil((float) num_chunks/block_size);

  // executes kernel
  blockSgmKernel<<<num_blocks, block_size>>>(block_size,
                                             num_chunks,
                                             sgm_offset,
                                             block_sgm);
  cudaThreadSynchronize();

}

// @summary: Wrapper for histogram kernel
template <class T>
void histogramConstructor(unsigned int block_size,
                          unsigned int   tot_size,
                          T*              input_h,
                          int*             hist_h){
  const unsigned int num_blocks   = ceil((float)tot_size/block_size);
  const unsigned int num_chunks   = ceil((float)tot_size/(CHUNK_SIZE*block_size));
  const unsigned int num_segments = ceil((float)tot_size/CHUNK_SIZE);
  //const unsigned int work_size  = ceil((float)tot_size/(CHUNK_SIZE*block_size));

  /* device variables */
  T*       d_in;
  int*   inds_d;
  int*   sorted_inds_d;
  int*   hist_d;
  int*   sgm_offset;
  int*   sgm_id_arr;
    
  cudaMalloc(  (void**)&d_in, tot_size * sizeof(T));
  cudaMalloc((void**)&inds_d, tot_size * sizeof(int));
  cudaMalloc((void**)&sorted_inds_d, tot_size * sizeof(int));
  cudaMemcpy(d_in, input_h, tot_size * sizeof(T), cudaMemcpyHostToDevice);

  // converts values to histogram indices
  histVals2IndexDevice<T>(tot_size, d_in, inds_d);

  // Free input array after converting to histogram indices
  cudaFree(d_in);

  // Sorts histogram index array
  radixSortDevice(inds_d, sorted_inds_d, tot_size);
  //cudaFree(inds_d);

  // allocate arrays to contain segment offset
  cudaMalloc((void**)&sgm_offset, num_segments*sizeof(int));
  
  // Find segment size, 
  // TODO: USE FUNCTION TO GET SEGMENT OFFSET ARRAY
  // WAITING ON JOACHIM TO COMMIT HIS REDUCTION TO FIND SEGMENT SHAPE ARRAY
  /* histVals2IndexDevice<int>(tot_size, sorted_inds_d, inds_d); */
  /* naiveHistKernel<<<num_blocks, block_size>>>(); */

  // allocates memory for segment id array
  cudaMalloc((void**)&sgm_id_arr, num_chunks*sizeof(T));

  /* // Find each blocks starting segment indices */
  /* blockSgm(block_size, tot_size, sgm_offset, sgm_id_arr); */

  // Allocates histogram on device
  cudaMalloc((void**)&hist_d, HISTOGRAM_SIZE*sizeof(int));

  /* // Constructs histogram, only works on histograms big enough to fit on GPU memory */
  /* grymersHistKernel<<<num_blocks, block_size>>>(tot_size,    // data input size */
  /*                                               num_chunks,  // number of workloads */
  /*                                               num_sgms,    // number of segments */
  /*                                               sgm_id_arr,  // Workload segment index */
  /*                                               sgm_offset,  // Segment offset array */
  /*                                               sorted_inds_d,      // histogram indexes */
  /*                                               hist_d);     // global histogram */
  /* cudaThreadSynchronize(); */
  
  /* // copying final histogram back to host from device */
  /* cudaMemcpy(hist_h, hist_d, HISTOGRAM_SIZE*sizeof(int), cudaMemcpyDeviceToHost); */

  // Free up all device memory
  cudaFree(inds_d);
  cudaFree(hist_d);
  cudaFree(sgm_offset);
  cudaFree(sgm_id_arr);
}


#endif //HOST_HIST
