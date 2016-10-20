#ifndef HOST_HIST
#define HOST_HIST

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <../cub/cub.cuh>
#include "Kernels.cu.h"
#include "setup.cu.h"

// Finds least and most significant bit
#define BEGIN_BIT                ceil(log2((float) CHUNK_SIZE))
#define END_BIT   max(BEGIN_BIT, ceil(log2((float) HISTOGRAM_SIZE)))

// A function which performs addition,
template<class T>
class Addition {
 public:
  typedef T BaseType;
  static __device__ __host__ inline T identity()                    { return (T)0;    }
  static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }
};

// Maps ...
template <class T>
void histIndex (unsigned int      d_size,
                unsigned int d_hist_size,
                unsigned int  block_size,
                T               boundary,
                T*                  d_in,
                T*                 d_out) {
  int num_blocks = ceil(d_size / block_size);
}

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

  // Figure out how much tempoary storage is needed
  cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_keys_out,
                                 array_length,
                                 BEGIN_BIT,
                                 END_BIT);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Call the library function
  cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_keys_out,
                                 array_length,
                                 BEGIN_BIT,
                                 END_BIT);

  // Write back the result
  cudaMemcpy(array_to_be_sorted, d_keys_out, sizeof(int)*array_length,
             cudaMemcpyDeviceToHost);

  // Clean up memory
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
}

#endif //HOST_HIST

