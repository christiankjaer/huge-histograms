#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Host.cu.h"
//#include "sequential/arraylib.cu.h"
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
void sortTest(unsigned int* inds, int data_size){
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
  float  max_rand_num1 = 1024*16;
  float  max_rand_num2 = 0.0;
  float* data          = (float*)malloc(data_size * sizeof(float));
  float* data_d;
  unsigned int* inds_seq = (unsigned int*)malloc(data_size * sizeof(int));
  unsigned int* inds_par = (unsigned int*)malloc(data_size * sizeof(int));

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
  compareTest<unsigned int>(inds_par, inds_seq, data_size);
  update();

  // clean up memory
  //free(data);
  cudaFree(data_d);
  free(inds_seq);
  free(inds_par);
  printf("\n");
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

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
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  printf("TESTING METADATA COMPUTATIONS\n");
  // max_rand_num1  = 500000.0;
  // data_size      = 50000;
  data           = (float*)malloc(data_size * sizeof(float));
  inds_seq       = (unsigned int*)malloc(data_size * sizeof(int));
  inds_par       = (unsigned int*)malloc(data_size * sizeof(int));
  unsigned int* inds_tmp  = (unsigned int*)malloc(data_size * sizeof(int));

  // generate test data.
  randArrSeq(data, data_size, max_rand_num1);
  arr2HistIdxSeq(data, inds_seq, data_size, max_rand_num1);
  radixSort(inds_seq, data_size);
  copyArray(inds_seq, inds_par, data_size);

  // test that the index_arrays are sorted and identical
  // otherwise, the rest of the test is invalid.
  sortTest(inds_seq, data_size);
  update();
  sortTest(inds_par, data_size);
  update();
  compareTest<unsigned int>(inds_par, inds_seq, data_size);
  update();

  // define known meta data about segments
  unsigned int  num_segments   = ceil(HISTOGRAM_SIZE / (float)CHUNK_SIZE);
  unsigned int* segment_sizes  = (unsigned int*)malloc(num_segments*sizeof(int));
  unsigned int* segment_sizes2 = (unsigned int*)malloc(num_segments*sizeof(int));
  int           chunk_size     = ceil((float)data_size/HARDWARE_PARALLELISM);
  int           block_workload = chunk_size * CUDA_BLOCK_SIZE;
  int           num_blocks     = ceil((float)data_size / block_workload);
  unsigned int* segment_sizes_d;
  unsigned int* block_sgm_index_d;
  unsigned int* segment_d;
  unsigned int* inds_d;
  cudaMalloc((void**)&segment_sizes_d, num_segments * sizeof(int));
  cudaMalloc((void**)&block_sgm_index_d, num_blocks*sizeof(int));
  cudaMalloc((void**)&segment_d, data_size * sizeof(int));
  cudaMalloc((void**)&inds_d, data_size * sizeof(int));
  cudaMemcpy(inds_d, inds_par, data_size * sizeof(int), cudaMemcpyHostToDevice);

  // Compare the results by visual inspection
  segmentSizesSeq(inds_seq, data_size, segment_sizes, num_segments);
  metaData(data_size, inds_d, num_segments, block_workload, segment_sizes_d, block_sgm_index_d);

  cudaMemcpy(segment_sizes2, segment_sizes_d,
             num_segments * sizeof(int), cudaMemcpyDeviceToHost);
  printf("\n");

  // TODO : Automate this comparison.
  printf("Segment Sizes\n");
  printIntArraySeq((int *) segment_sizes , num_segments);
  printf("Segment Offsets\n");
  printIntArraySeq((int *) segment_sizes2, num_segments);

  unsigned int* block_sgm_index = (unsigned int*)malloc(num_blocks*sizeof(int));
  cudaMemcpy(block_sgm_index, block_sgm_index_d,
             num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

  printf("Block segment id\n");
  printIntArraySeq((int *)block_sgm_index, num_blocks);

  // Test the sequential and parallel segmentsize funcitons
  printf("\n");
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  //update();

  // Clean up memory
  free(block_sgm_index);
  cudaFree(block_sgm_index_d);
  cudaFree(segment_sizes_d);
  cudaFree(segment_d);
  cudaFree(inds_d);
  cudaFree(data_d);
  free(segment_sizes);
  free(segment_sizes2);
  free(inds_par);
  free(inds_seq);
  free(inds_tmp);

  unsigned int* hist_h = (unsigned int*)malloc(HISTOGRAM_SIZE*sizeof(int));
  
  histogramConstructor<float>(data_size, data, hist_h);

  printf("Final histogram kernel\n");
  printIntArraySeq((int *) hist_h, HISTOGRAM_SIZE);
  free(hist_h);
  free(data);

  printf("TEST RESULTS\nPASSED: %d FAILED: %d.\n", passed, failed);
  return 0;
}
