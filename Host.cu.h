#ifndef HOST_HIST
#define HOST_HIST

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../cub/cub.cuh"
#include "Kernels.cu.h"
#include "setup.cu.h"

// @class : The addition function, where e = 0 and apply(t1, t2) = t1 + t2
template<class T>
class Addition {
 public:
  typedef T BaseType;
  static __device__ __host__ inline T identity()                    { return (T)0;    }
  static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }
};

// @class   : The addition function, where e = 0 and apply(t1, t2) = max(t1, t2)
// @remarks : The neutral element e = 0 comes from the assertion that
//            all values in the image are non-negative
template<class T>
class Maximum {
 public:
  typedef T BaseType;
  static __device__ __host__ inline T identity() { return (T)0; }
  static __device__ __host__ inline T apply(const T t1, const T t2)
  { return max(t1, t2); }
};

template<class OP, class T>
void scanInc(unsigned long arr_size,
             T*            arr_in,
             T*            arr_out) {

  unsigned int num_blocks  = ceil(arr_size / (float)CUDA_BLOCK_SIZE);
  unsigned int sh_mem_size = CUDA_BLOCK_SIZE * 32; // or just all of it !

  T* d_in;
  T* d_out;
  cudaMalloc((void**)&d_in,  arr_size * sizeof(T));
  cudaMalloc((void**)&d_out, arr_size * sizeof(T));
  cudaMemcpy(d_in, arr_in, arr_size * sizeof(T), cudaMemcpyHostToDevice);

  scanIncKernel<OP,T><<< num_blocks, CUDA_BLOCK_SIZE, sh_mem_size >>>
    (d_in, d_out, arr_size);
  cudaThreadSynchronize();

  // BASE CASE :
  // The reduction could fit into one cuda block, and we are done.

  if (CUDA_BLOCK_SIZE >= arr_size) {
    cudaMemcpy(arr_out, d_out, arr_size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  // RECURSIVE CASE:
  // we copy the end of each CUDA_BLOCK_SIZE blocks, and scan again !

  // Allocate new device input & output array of size num_blocks
  T* d_rec_in;
  T* d_rec_out;
  cudaMalloc((void**)&d_rec_in , num_blocks*sizeof(T));
  cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T));

  unsigned int num_blocks_rec = ceil(num_blocks / (float)CUDA_BLOCK_SIZE);

  // Copy in the end-of-block results of the previous scan
  copyEndOfBlockKernel<T><<< num_blocks_rec, CUDA_BLOCK_SIZE >>>
    (d_out, d_rec_in, num_blocks);
  cudaThreadSynchronize();

  // Scan recursively the last elements of each CUDA block
  scanInc<OP, T>(num_blocks, d_rec_in, d_rec_out);

  // Distribute the the corresponding element of the
  // recursively scanned data to all elements of the
  // corresponding original block
  distributeEndBlock<OP, T><<< num_blocks, CUDA_BLOCK_SIZE >>>
    (d_rec_out, d_out, arr_size);
  cudaThreadSynchronize();

  // Copy back the result of the scan
  cudaMemcpy(arr_out, d_out, arr_size * sizeof(T), cudaMemcpyDeviceToHost);

  // Clean up memory.
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_rec_in );
  cudaFree(d_rec_out);
}

// @summary : computes the maximum value in an array of values of type T.
template<class T>
T maximumElement(T* arr, int arr_size){

  // Scan with the maximum class.
  T* output = (T*)malloc(arr_size * sizeof(T));
  scanInc<Maximum<float>, float>(arr_size, arr, output);

  // Pick the last element
  T  result = output[arr_size-1];

  // Clean up memory
  free(output);
  return result;
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

  int num_blocks = ceil(arr_size / CUDA_BLOCK_SIZE);
  T boundary = maximumElement<T>(vals_h, arr_size);

  // Allocate device memory
  T*   vals_d;
  int* inds_d;
  cudaMalloc((void**)&vals_d, arr_size * sizeof(float));
  cudaMalloc((void**)&inds_d, arr_size * sizeof(int));
  cudaMemcpy(vals_d, vals_h, arr_size*sizeof(T), cudaMemcpyHostToDevice);

  // TODO : handle {arr_size > sizeof(shared_memory)} !

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

