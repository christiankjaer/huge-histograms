#ifndef HOST_HIST
#define HOST_HIST

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../cub/cub.cuh"
#include "Kernels.cu.h"
#include "setup.cu.h"

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
  int end_bit   = sizeof(int)*8;
  /* int end_bit   = max(begin_bit, (int)ceil(log2((float) HISTOGRAM_SIZE))) + 1; */

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

#endif //HOST_HIST

