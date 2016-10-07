#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cub/cub.cuh>

void fill_array(int* array, int array_length){
  for (int i = 0; i < array_length; i++){
    array[i] = (int)rand();
  }
}

__inline__ void make_keys(int* A, int* K, int N, int MASK){
  for (int i = 0; i < N; i++){
    K[i] = A[i] & MASK;
  }
}

void print_array(int* array, int array_length){
  printf("[");
  for (int i = 0; i < array_length; i++){
    printf("0x%8x\n", array[i]);
    if (i != array_length-1){
      printf(", ");
    }
  }
  printf("].\n");
}

int main (int args, char** argv){
  printf("THIS PROGRAM TRIES TO SORT A HUGE ARRAY\n");
  int  array_length        = 32;
  int  begin_bit           = sizeof(int)*4;
  int  end_bit             = sizeof(int)*8;
  int* array_to_be_sorted  = (int*)malloc(array_length * sizeof(int));
  int* d_keys_in;
  int* d_keys_out;
  cudaMalloc(&d_keys_in, sizeof(int)*array_length);
  cudaMalloc(&d_keys_out, sizeof(int)*array_length);
  fill_array(array_to_be_sorted, array_length);
  cudaMemcpy(d_keys_in, array_to_be_sorted, sizeof(int)*array_length,
             cudaMemcpyHostToDevice);

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_keys_out,
                                 array_length,
                                 begin_bit,
                                 end_bit);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_keys_out,
                                 array_length,
                                 begin_bit,
                                 end_bit);

  cudaMemcpy(array_to_be_sorted, d_keys_out, sizeof(int)*array_length,
             cudaMemcpyDeviceToHost);

  print_array(array_to_be_sorted, array_length);
  // make_keys(array_to_be_sorted, array_keys, array_length, key_mask);
  // print_array(array_keys, array_length);

  free(array_to_be_sorted);
  return 0;
}
