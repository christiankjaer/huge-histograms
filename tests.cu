#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Host.cu.h"
#include "sequential/arraylib.cu.h"
#include "setup.cu.h"

int  passed = 0;
int  failed = 0;
bool result = true;

// @summary : book keeper, to be called between tests.
void update(){
  if (result){
    passed++;
    printf(".");
  }
  else{
    failed++;
    printf("*");
    result = true;
  }
}

// @summary : You know what this does {~_^}
bool myAssert(bool test){
  result = result && test;
  return result;
}

// @test : asserts the inds array has been partially sorted.
//         in segments of values varying with at most CHUNK_SIZE
void sortTest(int* inds, int data_size){
  int _min = 0;
  int _max = CHUNK_SIZE;
  for (int i = 0; i < data_size; i++){
    while (inds[i] >= _max){
      _min += CHUNK_SIZE;
      _max += CHUNK_SIZE;
    }
   if (!myAssert(_min <= inds[i] && inds[i] < _max)) return;
  }
}

// @test    : compares two arrays of type T are exactly the same.
// @remarks : assumes the two arrays to be of equal size.
template<class T>
void compareTest(T* result, T* expected, int size){
  for(int i = 0; i < size; i++){
    if (!myAssert(result[i] == expected[i])) return;
  }
}

// @test    : compares two arrays of type T differ with at most epsilon.
// @remarks : assumes the two arrays to be of equal size.
template<class T>
void compareTestEps(T* result, T* expected, int size, T eps){
  for(int i = 0; i < size; i++){
    if (!myAssert(abs(result[i] - expected[i]) < eps)) return;
  }
}


// @summary : running the above test functions.
int main (int args, char** argv){

  printf("TESTING RADIX SORT\n");

  // declare initial values
  int    data_size     = 1024*16;
  float  max_rand_num1 = 30000.0;
  float  max_rand_num2 = 0.0;
  float* data          = (float*)malloc(data_size * sizeof(float));
  float* data_d;
  int*   inds_seq      = (int*)malloc(data_size * sizeof(int));
  int*   inds_par      = (int*)malloc(data_size * sizeof(int));

  // fill the data array with random values
  randArrSeq(data, data_size, max_rand_num1);

  // check that the maximum number was found correctly
  cudaMalloc((void**)&data_d, data_size * sizeof(float));
  cudaMemcpy(data_d, data, data_size * sizeof(float), cudaMemcpyHostToDevice);
  max_rand_num1 = maximumElementSeq(data, data_size);
  max_rand_num2 = maximumElement<float>(data_d, data_size);
  myAssert(max_rand_num1 == max_rand_num2);
  update();

  // generate and sort the index values sequentially
  arr2HistIdxSeq(data, inds_seq, data_size, max_rand_num1);
  radixSort(inds_seq, data_size);

  // generate and sort the index values in parallel
  histVals2Index<float>(data_size, data, inds_par);
  radixSort(inds_par, data_size);

  // test that the partial sorting is korrekt
  sortTest(inds_seq, data_size);
  update();
  sortTest(inds_par, data_size);
  update();

  // ensure that the two index arrays match
  compareTest<int>(inds_par, inds_seq, data_size);
  update();

  // clean up memory
  free(data);
  cudaFree(data_d);
  free(inds_seq);
  free(inds_par);
  printf("\n");

  printf("TESTING PREFIX SUM EXCLUSIVE\n");

  // declare initial values
  max_rand_num1  = 100.0;
  data           = (float*)malloc(data_size * sizeof(float));
  float  epsilon = 0.001;
  float* data_out;
  float* prefix_sum_seq = (float*)malloc(data_size * sizeof(float));
  float* prefix_sum_par = (float*)malloc(data_size * sizeof(float));

  // fill the data array with random values
  randArrSeq(data, data_size, max_rand_num1);

  // copy values to device
  cudaMalloc((void**)&data_d, data_size * sizeof(float));
  cudaMalloc((void**)&data_out, data_size * sizeof(float));
  cudaMemcpy(data_d, data, data_size * sizeof(float), cudaMemcpyHostToDevice);

  // compute the two prefix sum exclusives.
  prefixSumExc(data_size, data_d, data_out);
  scanExcPlusSeq(data, prefix_sum_seq, data_size);

  // copy back the result from device
  cudaMemcpy(prefix_sum_par, data_out, data_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  // check that the resulting arrays differ with at most epsilon.
  compareTestEps<float>(prefix_sum_par, prefix_sum_seq, data_size, epsilon);
  update();

  // clean up memory
  cudaFree(data_d);
  cudaFree(data_out);
  free(data);
  free(prefix_sum_par);
  free(prefix_sum_seq);
  printf("\n");

  printf("TESTING METADATA COMPUTATIONS\n");
  max_rand_num1  = 500000.0;
  data_size      = 5000;
  data           = (float*)malloc(data_size * sizeof(float));
  inds_seq       = (int*)malloc(data_size * sizeof(int));
  int* inds_tmp  = (int*)malloc(data_size * sizeof(int));

  // fill the data array with random values and genreate indexes
  randArrSeq(data, data_size, max_rand_num1);
  arr2HistIdxSeq(data, inds_seq, data_size, max_rand_num1);

  // varify that the sorting has happend
  // (otherwise, the next test does not make sense)
  radixSort(inds_seq, data_size);
  sortTest(inds_seq, data_size);
  update();

  int  num_segments = ceil(HISTOGRAM_SIZE / (float)CHUNK_SIZE);
  int* segment_sizes = (int*)malloc(num_segments*sizeof(int));
  segmentSizesSeq(inds_seq, data_size, segment_sizes, num_segments);

  printIntArraySeq(segment_sizes, num_segments);
  // TODO : Write a test, which matches with the parallel function

  // test the sequential and parallel segmentsize funcitons
  printf("\n");

  free(inds_seq);
  free(inds_tmp);
  free(data);

  printf("TEST RESULTS\nPASSED: %d FAILED: %d.\n", passed, failed);
  return 0;
}
