#ifndef HOST_HIST
#define HOST_HIST

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cub/cub.cuh>
#include "sequential/arraylib.cu.h"

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
                           unsigned int      hist_size,
                           unsigned int*     inds_d){

  // Allocate device memory
  /* T*   vals_d; */
  /* int* inds_d; */
  /* cudaMalloc((void**)&vals_d, arr_size * sizeof(float)); */
  /* cudaMalloc((void**)&inds_d, arr_size * sizeof(int)); */
  /* cudaMemcpy(vals_d, vals_h, arr_size*sizeof(T), cudaMemcpyHostToDevice); */

  //  Figure out the boundaries (vague).
  int num_blocks = ceil((float)arr_size / CUDA_BLOCK_SIZE);
  T   boundary   = maximumElement<T>(vals_d, arr_size);

  // TODO : handle {arr_size > sizeof(shared_memory)} !?

  // Run indexing kernel
  histVals2IndexKernel<T><<<num_blocks, CUDA_BLOCK_SIZE>>>
    (vals_d, inds_d, arr_size, hist_size, boundary);
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
                     unsigned int*     inds_h){

  // Allocate device memory
  T*   vals_d;
  unsigned int* inds_d;
  cudaMalloc((void**)&vals_d, arr_size * sizeof(float));
  cudaMalloc((void**)&inds_d, arr_size * sizeof(int));
  cudaMemcpy(vals_d, vals_h, arr_size*sizeof(T), cudaMemcpyHostToDevice);

  //  Figure out the boundaries (vague).
  int num_blocks = ceil((float)arr_size / CUDA_BLOCK_SIZE);
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
void radixSortDevice(unsigned int* unsorted_d,
                     unsigned int* sorted_d,
                     unsigned int  array_length){

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
void radixSort(unsigned int* array_to_be_sorted,
               unsigned int  array_length){

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


// @summary : for now, just computes segment_sizes.
void metaData(unsigned int  inds_size,
              unsigned int* inds_d,
              unsigned int  num_segments,
              unsigned int  block_workload,
              unsigned int* segment_sizes_d,
              unsigned int* block_sgm_index_d
              ){
  int num_blocks = ceil((float)inds_size / CUDA_BLOCK_SIZE);
  cudaMemset(segment_sizes_d, 0, num_segments * sizeof(unsigned int));
  segmentMetaData<<<num_blocks, CUDA_BLOCK_SIZE>>>
    (inds_d,
     inds_size,
     segment_sizes_d,
     block_workload,
     CUDA_BLOCK_SIZE,
     block_sgm_index_d);
  cudaThreadSynchronize();
}

template <class T>
void largeHistogram(unsigned int image_size,
                    T* d_image,
                    unsigned int histogram_size,
                    unsigned int* d_hist) {


  unsigned int chunk_size = ceil((float)image_size / HARDWARE_PARALLELISM);
  unsigned int block_workload = chunk_size * CUDA_BLOCK_SIZE;
  unsigned int num_blocks = ceil((float)image_size / block_workload);
  unsigned int num_segments = ceil((float)histogram_size / GPU_HIST_SIZE);

  unsigned int *d_inds, *d_sgm_idx, *d_sgm_offset, *d_sorted;

  cudaMalloc(&d_inds, sizeof(unsigned int)*image_size);
  cudaMalloc(&d_sorted, sizeof(unsigned int)*image_size);
  cudaMalloc(&d_sgm_idx, sizeof(unsigned int)*num_blocks);
  cudaMalloc(&d_sgm_offset, sizeof(unsigned int)*num_segments);

  histVals2IndexDevice<float>(image_size, d_image, histogram_size, d_inds);
  radixSortDevice(d_inds, d_sorted, image_size);
  metaData(image_size, d_sorted, num_segments, block_workload, d_sgm_offset, d_sgm_idx);

  cudaMemset(d_hist, 0, sizeof(unsigned int)*histogram_size);

  christiansHistKernel<<<num_blocks, CUDA_BLOCK_SIZE>>>
    (image_size, histogram_size, chunk_size, d_sgm_idx, d_sgm_offset, d_sorted, d_hist);

  cudaThreadSynchronize();

  cudaFree(d_inds);
  cudaFree(d_sgm_idx);
  cudaFree(d_sgm_offset);
  cudaFree(d_sorted);

}

// @summary: Wrapper for histogram kernel
template <class T>
void histogramConstructor(unsigned int   tot_size,
                          T*              input_h,
                          unsigned int*    hist_h){
  const unsigned int thread_workload   = ceil((float)tot_size/HARDWARE_PARALLELISM);
  const unsigned int block_workload    = ceil((float)CUDA_BLOCK_SIZE*thread_workload);
  const unsigned int num_blocks        = ceil((float)tot_size/block_workload);
  const unsigned int num_segments      = ceil((float)HISTOGRAM_SIZE/CHUNK_SIZE);
  //const unsigned int work_size  = ceil((float)tot_size/(CHUNK_SIZE*block_size));
  printf("tw: %d, bw %d, nb %d, ns %d\n", 
         thread_workload,
         block_workload,
         num_blocks,
         num_segments);


  /* device variables */
  T*                       d_in;
  unsigned int*          inds_d;
  unsigned int*   sorted_inds_d;
  unsigned int*          hist_d;
  unsigned int*      sgm_offset;
  unsigned int*    block_sgm_id;
  //unsigned int*            test;

  cudaMalloc(  (void**)&d_in, tot_size * sizeof(T));
  cudaMalloc((void**)&inds_d, tot_size * sizeof(int));
  cudaMemcpy(d_in, input_h, tot_size * sizeof(T), cudaMemcpyHostToDevice);

  // converts values to histogram indices
  histVals2IndexDevice<T>(tot_size, d_in, inds_d);

  /* test = (unsigned int*)malloc(tot_size*sizeof(int)); */
  /* cudaMemcpy(test, inds_d, tot_size * sizeof(T), cudaMemcpyDeviceToHost); */
  /* printf("vals to inds array \n"); */
  /* printIntArraySeq((int*) test, tot_size); */
  /* free(test); */
  // Free input array after converting to histogram indices
  cudaFree(d_in);

  cudaMalloc((void**)&sorted_inds_d, tot_size * sizeof(int));
  // Sorts histogram index array
  radixSortDevice(inds_d, sorted_inds_d, tot_size);

  /* test = (unsigned int*)malloc(tot_size*sizeof(int)); */
  /* cudaMemcpy(test, sorted_inds_d, tot_size * sizeof(T), cudaMemcpyDeviceToHost); */
  /* printf("sorted inds array \n"); */
  /* printIntArraySeq((int*) test, tot_size); */
  /* free(test); */

  //free random hist inds;
  cudaFree(inds_d);

  // allocate arrays to contain segment offset, and block segment index
  cudaMalloc((void**)&sgm_offset, num_segments*sizeof(int));
  cudaMalloc((void**)&block_sgm_id, num_blocks  *sizeof(int));
  /* cudaMemset(sgm_offset, 0, num_segments * sizeof(int)); */
  /* cudaMemset(block_sgm_id, 0, num_blocks * sizeof(int)); */
  // Find segment offset and segment for which a block starts in
  // Meta data computes both
  metaData(tot_size,
           sorted_inds_d,
           num_segments,
           block_workload,
           sgm_offset,
           block_sgm_id);

  /* test = (unsigned int*)malloc(num_segments*sizeof(int)); */
  /* cudaMemcpy(test, sgm_offset, num_segments * sizeof(T), cudaMemcpyDeviceToHost); */
  /* printf("segment offsets \n"); */
  /* printIntArraySeq((int*) test, num_segments); */
  /* free(test); */

  /* test = (unsigned int*)malloc(num_segments*sizeof(int)); */
  /* cudaMemcpy(test, block_sgm_id, num_blocks * sizeof(T), cudaMemcpyDeviceToHost); */
  /* printf("block segment id array \n"); */
  /* printIntArraySeq((int*) test, num_blocks); */
  /* free(test); */

  // Allocates histogram on device
  cudaMalloc((void**)&hist_d, HISTOGRAM_SIZE*sizeof(int));
  cudaMemset(hist_d, 0, HISTOGRAM_SIZE * sizeof(unsigned int));

  // Constructs histogram, only works on histograms big enough to fit on GPU memory
  christiansHistKernel<<<num_blocks, CUDA_BLOCK_SIZE>>>(tot_size,
                                                        HISTOGRAM_SIZE,
                                                        thread_workload,
                                                        block_sgm_id,
                                                        sgm_offset,
                                                        sorted_inds_d,
                                                        hist_d);
  /* grymersHistKernel<<<num_blocks, CUDA_BLOCK_SIZE>>>(tot_size,       // data input size */
  /*                                                    block_workload, // number of workloads */
  /*                                                    num_segments,   // number of segments */
  /*                                                    block_sgm_id,   // Workload segment index */
  /*                                                    sgm_offset,     // Segment offset array */
  /*                                                    sorted_inds_d,  // histogram indexes */
  /*                                                    hist_d);        // global histogram */
  cudaThreadSynchronize();

  // copying final histogram back to host from device
  cudaMemcpy(hist_h, hist_d, HISTOGRAM_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

  // Free up all device memory
  cudaFree(inds_d);
  cudaFree(hist_d);
  cudaFree(sgm_offset);
  cudaFree(block_sgm_id);
}


#endif //HOST_HIST
