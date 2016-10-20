// Here's a small library for seqential functions on arrays.

#ifndef ARRAY_LIB
#include "../setup.cu.h"

// @summary : maps data to histogram indexes.
// @params  : input_arr -> data array
//            hist_inds -> target array
//            size_arr  -> input array size
//            size_hist -> the histogram size
void arr2HistIdxSeq(float* input_arr, int* hist_inds,
                    int size_arr, float max_input) {
  for (int i=0; i < size_arr; i++) {
    hist_inds[i] = (int)((input_arr[i]/max_input)*(float)HISTOGRAM_SIZE);
  }
}

template<class T>
T maximumElementSeq(T* array, arr_size){
  T my_max = array[0];
  for (int i = 1; i < arr_size; i++){
    my_max = max (my_max, array[i]);
  }
}


// @summary : fills an array with random floats
void randArrSeq(float* array, int array_length, float max_rand_num){
  for (int i = 0; i < array_length; i++){
    array[i] = (float)(rand() % (int)max_rand_num);
  }
}

// @summary : fills an array with zeroes
void zeroArrSeq(float* array, int array_length){
  for (int i = 0; i < array_length; i++){
    array[i] = 0.0;
  }
}

// @summary : a sequential implementation of scanExc (+)
void scanExcSeq(float* array, int array_length){
  for (int i = 1; i < array_length; i++){
    array[i] = array[i-1];
  }
  array[0] = 0;
  for (int i = 1; i < array_length; i++){
    array[i] = array[i-1] + array[i];
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
void printIntArraySeq(int* array, int array_length){
  printf("[");
  int j = 0;
  for (int i = 0; i < array_length; i++){
    printf("%6d", array[i]);
    if (i != array_length-1){
      printf(",");
      if (j == 10) {printf("\n "); j = 0;}
      else{j++;}
    }
  }
  printf("].\n");
}

// @summary : prints an array to terminal for visual inspection
// @remarks : prints only 10 elements pr. line
void printFloatArraySeq(float* array, int array_length){
  printf("[");
  int j = 0;
  for (int i = 0; i < array_length; i++){
    printf("%6f", array[i]);
    if (i != array_length-1){
      printf(",");
      if (j == 10) {printf("\n "); j = 0;}
      else{j++;}
    }
  }
  printf("].\n");
}

#endif //ARRAY_LIB
