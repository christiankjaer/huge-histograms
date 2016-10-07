#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "Host.cu.h"

void fill_array(int* array, int array_length){
  for (int i = 0; i < array_length; i++){
    array[i] = (int)rand() % 30000;
  }
}

__inline__ void make_keys(int* A, int* K, int N, int MASK){
  for (int i = 0; i < N; i++){
    K[i] = A[i] & MASK;
  }
}

void print_array(int* array, int array_length){
  printf("[");
  int j = 0;
  for (int i = 0; i < array_length; i++){
    printf("%d", array[i]);
    if (i != array_length-1){
      printf(",");
      j = j;
      printf("\n");
    }
  }
  printf("].\n");
}


int main (int args, char** argv){
  printf("TESTING RADIX SORT\n");
  int  array_length        = 32;
  int* array_to_be_sorted  = (int*)malloc(array_length * sizeof(int));

  fill_array(array_to_be_sorted, array_length);
  histogram_radix_sort(array_to_be_sorted, array_length);
  print_array(array_to_be_sorted,array_length);


  free(array_to_be_sorted);
  return 0;
}
