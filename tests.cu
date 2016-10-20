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
  int    data_size    = 1024*32;
  float  max_rand_num = 30000.0;
  float* data         = (float*)malloc(data_size * sizeof(float));
  int*   inds_seq     = (int*)malloc(data_size * sizeof(int));
  int*   inds_par     = (int*)malloc(data_size * sizeof(int));
  randArrSeq(data, data_size, max_rand_num);
  arr2HistIdxSeq(data, inds_seq, data_size, max_rand_num);
  radixSort(inds_seq, data_size);
  sortTest(inds_seq, data_size);
  update();
  histVals2Index<float>(data_size, max_rand_num, data, inds_par);
  radixSort(inds_par, data_size);
  sortTest(inds_par, data_size);
  update();
  compareTest<int>(inds_par, inds_seq, data_size);
  update();
  free(data);
  free(inds_seq);
  free(inds_par);
  printf("\n");


  printf("TEST RESULTS\nPASSED: %d FAILED: %d.\n", passed, failed);
  return 0;
}