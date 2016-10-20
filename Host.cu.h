#ifndef HOST_HIST
#define HOST_HIST
#define GPU_HISTOGRAM_SIZE 8192

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cub/cub.cuh>
#include "Kernels.cu.h"


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

  mapIndKernel<T>
    <<<num_blocks, block_size>>>(d_size, d_hist_size, boundary, d_in, d_out);
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

  int    begin_bit = ceil(log2((float)GPU_HISTOGRAM_SIZE));
  int    end_bit   = sizeof(int)*8; // TODO, write a more intelligent version

  void   *d_temp_storage    = NULL;
  size_t temp_storage_bytes = 0;

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

