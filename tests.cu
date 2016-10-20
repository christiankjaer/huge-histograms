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
void myAssert(bool test){
  result = result && test;
}

// @test : asserts the inds array has been partially sorted.
void sortTest0(int* inds, int data_size){
  int _min = 0;
  int _max = CHUNCK_SIZE;
  for (int i = 0; i < data_size; i++){
    while (inds[i] >= _max){
      _min += CHUNCK_SIZE;
      _max += CHUNCK_SIZE;
    }
    myAssert(_min <= inds[i] && inds[i] < _max);
  }
}

// @summary : running the above test functions.
int main (int args, char** argv){

  printf("TESTING RADIX SORT\n");
  int    data_size = 1024*32;
  int    hist_size = 1024*8;
  float* data = (float*)malloc(data_size * sizeof(float));
  int*   inds = (int*)malloc(data_size * sizeof(int));
  randArrSeq(data, data_size);
  arr2HistIdxSeq(data, inds, data_size, hist_size, MAX_RAND_NUM);
  radixSort(inds, data_size);
  sortTest0(inds, data_size);
  update();
  free(data);
  free(inds);
  printf("\n");

  printf("TEST RESULTS\nPASSED: %d FAILED: %d.\n", passed, failed);
  return 0;
}
