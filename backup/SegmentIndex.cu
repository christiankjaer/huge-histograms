#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Host.cu.h"
#include "sequential/histCPU.cu.h"
#include "sequential/arraylib.cu.h"

// A way to compute histram indexes for the large/huge histogram
int main (){

  // Create input array (max val defined by MAX_RANDOM_NUMBER_SIZE)
  int  array_length = 1024;
  float* input_arr  = (float*)malloc(array_length * sizeof(float));
  fill_array(input_arr, array_length);

  // Convert to indices in a histogram
  int size_hist = 10000;
  int* hist_idx_arr = (int*)malloc(array_length * sizeof(int));
  arr_to_hist_idx(input_arr,hist_idx_arr,array_length,size_hist,MAX_RANDOM_NUMBER_SIZE);

  // Sort indices by the 13th bit (shared memory size for a block on GPU)
  radix_sort(hist_idx_arr,  array_length);
  print_array(hist_idx_arr, array_length);

  int  num_segments        = ceil((float)array_length / GPU_HISTOGRAM_SIZE);
  int* segment_offset_arr = (int*)malloc(num_segments * sizeof(int));

  zero_array(segment_offset_arr, num_segments);

  printf("num_segments %d\n", num_segments);

  // TODO : add a function which computes indexes.

  segmentSize(hist_idx_arr,
              segment_offset_arr,
              array_length);

  scan_exc(segment_offset_arr, num_segments);
  print_array(segment_offset_arr, num_segments);

  free(segment_offset_arr);
  free(input_arr);
  free(hist_idx_arr);
}
