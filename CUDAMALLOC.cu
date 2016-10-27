#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Host.cu.h"
#include "sequential/arraylib.cu.h"
#include "setup.cu.h"

int main (int args, char** argv){

  printf("%s", );

  float* big_memory_ptr;
  long int to_much_memory = 6000000000;
  cudaMalloc((void**)&big_memory_ptr, to_much_memory);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemset(big_memory_ptr, 0, to_much_memory);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaFree(big_memory_ptr);
  return 0;
}
