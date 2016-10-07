#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "Host.cu.h"
#include "sequential/arraylib.cu.h"



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
