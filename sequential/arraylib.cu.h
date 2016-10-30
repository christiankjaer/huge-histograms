// Here's a small library for seqential functions on arrays.

#ifndef ARRAY_LIB
#include "../setup.cu.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

// @summary : maps data to histogram indexes.
// @params  : input_arr -> data array
//            hist_inds -> target array
//            size_arr  -> input array size
//            size_hist -> the histogram size
void arr2HistIdxSeq(float* input_arr, unsigned int* hist_inds,
                    int size_arr, float max_input, unsigned int hist_size) {
  for (int i=0; i < size_arr; i++) {
    hist_inds[i] = (int)((input_arr[i]/max_input)*(float)hist_size);
  }
}

// @summary : counts the number of segments for the index array
// @returns : the array of segment sizes
void segmentSizesSeq(unsigned int* inds, int inds_size, 
                     unsigned int* segment_sizes, int num_segments){
  int this_segment_max = GPU_HIST_SIZE;
  int this_segment     = 0;
  // zero segment_sizes
  for (int i = 0; i < num_segments; i ++){
    segment_sizes[i] = 0;
  }
  // assumes the inds_array already partially sorted
  for (int i = 0; i < inds_size; i++){
    if (inds[i] >= this_segment_max){
      this_segment_max += GPU_HIST_SIZE;
      this_segment++;
    }
    segment_sizes[this_segment] += 1;
  }
}

// @summary : finds, the greatest element, in an array of floats
float maximumElementSeq(float* array, int arr_size){
  float my_max = array[0];
  for (int i = 1; i < arr_size; i++){
    my_max = max (my_max, array[i]);
  }
  return my_max;
}


// @summary : fills an array with random floats
void randArrSeq(float* array, int array_length, float max_rand_num){
  for (int i = 0; i < array_length; i++){
    array[i] = (float)(rand() % (int)max_rand_num);
  }
}

// @summary : fills an array with normal distirbuted floats.
// @remarks : dont free the array after ! (something is wrong).
void normArrSeq(float* array, int array_length, float max_rand_num){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(max_rand_num/2, max_rand_num/4);
  for (int i = 0; i < array_length; i++){
    array[i] = (float)min(max_rand_num, max(0.0, (std::round(d(gen)))));
  }
}

// @summary : fills an array with zeroes
template<class T>
void zeroArrSeq(T* array, int array_length){
  for (int i = 0; i < array_length; i++){
    array[i] = (T)0;
  }
}

// @summary : a sequential implementation of scanExc (+)
void scanExcPlusSeq(float* array, float* out, int array_length){
  out[0] = 0.0;
  for (int i = 1; i < array_length; i++){
    out[i] = out[i-1] + array[i-1];
  }
}

// @summary : makes a key set for raddix sort
// @prams   : A    -> input array ptr
//          : K    -> output key array
//          : N    -> array length (must be the same for A and K)
//          : MASK -> a mask value for the key indexes
__inline__ void makeKeysSeq(int* A, int* K, int N, int MASK){
  for (int i = 0; i < N; i++){
    K[i] = A[i] & MASK;
  }
}

// @summary : prints an array to terminal for visual inspection
// @remarks : prints only 10 elements pr. line
template<class T>
void printIntArraySeq(T* array, int array_length){
  printf("[");
  int j = 1;
  for (int i = 0; i < array_length; i++){
    printf("%6d", array[i]);
    if (i != array_length-1){
      printf(",");
      if (j == 10) {printf(" -- index %d\n", i); j = 1;}
      else{j++;}
    }
  }
  printf("].\n");
}

// @summary : prints an array to terminal for visual inspection
// @remarks : prints only 10 elements pr. line
void printFloatArraySeq(float* array, int array_length){
  printf("[");
  int j = 1;
  for (int i = 0; i < array_length; i++){
    printf("%6f", array[i]);
    if (i != array_length-1){
      printf(",");
      if (j == 10) {printf("\n "); j = 1;}
      else{j++;}
    }
  }
  printf("].\n");
}

template<class T>
void copyArray (T* arr, T* dst,int arr_size){
  for (int i = 0; i < arr_size; i++){
    dst[i] = arr[i];
  }
}

#endif //ARRAY_LIB
