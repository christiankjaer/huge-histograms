#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Host.cu.h"
#include "sequential/histCPU.cu.h"
#include "sequential/arraylib.cu.h"

// A way to compute histram indexes for the large/huge histogram
int main (){

  int  array_length = 1024;
  int* input_array  = (int*)malloc(array_length * sizeof(int));

  fill_array(input_array,  array_length);
  radix_sort(input_array,  array_length);
  print_array(input_array, array_length);

  int  num_segments        = ceil((float)array_length / GPU_HISTOGRAM_SIZE);
  int* segment_size_offset = (int*)malloc(num_segments * sizeof(int));

  zero_array(segment_size_offset, num_segments);

  printf("num_segments %d\n", num_segments);

  // TODO : add a function which computes indexes.

  int histogram_size = ceil((float)MAX_RANDOM_NUMBER_SIZE/8192);

  segmentSize(input_array,
              segment_size_offset,
              array_length,
              histogram_size);

  scan_exc(segment_size_offset, histogram_size);
  print_array(segment_size_offset, histogram_size);

  free(segment_size_offset);
  free(input_array);
}
