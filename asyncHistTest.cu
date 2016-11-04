#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "Kernels.cu.h"
#include "Host.cu.h"

#define IMG_SIZE 8192*768
#define HIST_SIZE 8192*12


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


template <class T>
void calc_indices(size_t N, size_t M, T *in, unsigned int *out, T boundary) {
  for (size_t i = 0; i < N; i++) {
    out[i] = (unsigned int) ((in[i] / boundary) * (T) (M - 1));
  }
}

void fill_histogram(size_t N, size_t M,
                    unsigned int *in, unsigned int *hist) {
  for (size_t i = 0; i < N; i++) {
    hist[in[i]]++;
  }
}

template <class T>
T max_elem(size_t arr_size, T* arr) {
  if (arr_size == 0) {
    return (T)0;
  }
  T maxe = arr[0];
  for (int i = 1; i < arr_size; i++) {
    maxe = arr[i] > maxe ? arr[i] : maxe;
  }
  return maxe;
}

// @test    : compares two arrays of type T are exactly the same.
// @remarks : assumes the two arrays to be of equal size.
template<class T>
void compareTest(T* result, T* expected, int size){
  for(int i = 0; i < size; i++){
    if (!myAssert(result[i] == expected[i])) return;
  }
}


template <class T>
void cpu_hist(size_t image_size, T* image, size_t hist_size, unsigned int* hist) {
  unsigned int *inds = (unsigned int*) malloc(sizeof(unsigned int[image_size]));
  memset(hist, 0, sizeof(unsigned int)*hist_size);
  calc_indices<T>(image_size, hist_size, image, inds, max_elem<T>(image_size, image));
  fill_histogram(image_size, hist_size, inds, hist);
}

// @test    : compares two arrays of type T differ with at most epsilon.
// @remarks : assumes the two arrays to be of equal size.
template<class T>
void compareTestEps(T* result, T* expected, int size, T eps){
  for(int i = 0; i < size; i++){
    if (!myAssert(abs(result[i] - expected[i]) < eps)) return;
  }
}

template<class T>
T sumSeq(T* arr, int arr_size){
  T sum = (T) 0;
  for (int i = 0; i < arr_size; i++){
    sum = sum + arr[i];
  }
  return sum;
}

template <class T>
void test_hist(unsigned int image_sz, unsigned int hist_sz) {

  printf("\nTesting histogram with\nimage size: %d\nhist size:%d\n", image_sz, hist_sz);
 
  T* data = (T*)malloc(image_sz * sizeof(T));
  T* out = (T*)malloc(image_sz * sizeof(T));
  T* out_async = (T*)malloc(image_sz * sizeof(T));
  unsigned int* hist = (unsigned int*) malloc(hist_sz*sizeof(int));
  unsigned int* hist_naive = (unsigned int*) malloc(hist_sz*sizeof(int));
  unsigned int* hist_cpu = (unsigned int*) malloc(hist_sz*sizeof(int));
  for (size_t i = 0; i < image_sz; i++) {
    data[i] = ((T)rand()/(T)RAND_MAX)*16;
    out[i] = (T) 0;
  }
  for (size_t i = 0; i < hist_sz; i++){
    hist[i] = 0;
    hist_naive[i] = 0;
  }
  
  // naiveSquare<T>(image_sz, data, out);
  // asyncSquare<T> (image_sz, data, out_async);
  
  //printFloatArraySeq(data, image_sz);

  T* d_data;
  unsigned int *d_hist_naive;
  cudaMalloc((void**)&d_data, image_sz*sizeof(T));
  cudaMalloc((void**)&d_hist_naive, hist_sz*sizeof(unsigned int));
  cudaMemcpy(d_data, data, image_sz*sizeof(T), cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_hist_naive, hist_naive, _sz*sizeof(T), cudaMemcpyHostToDevice);
  printf("HIST LARGE (GPU) in %d µs\n", largeHistogram<T>(image_sz, d_data, hist_sz, d_hist_naive));
  cudaMemcpy(hist_naive, d_hist_naive, hist_sz*sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_hist_naive);


  struct timeval t_start, t_end;
  unsigned long int elapsed;
  gettimeofday(&t_start, NULL);

  //cpu_hist(image_sz, data, hist_sz, hist_cpu);

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("histogram (CPU) in %d µs\n", elapsed);

  T max = maximumElement<T>(d_data, image_sz);
  printf("Max elem: %5.4f\n", max);
  cudaFree(d_data);

  asyncHist<T>(image_sz, data, hist_sz, hist, max);
  


  //compareTest<unsigned int>(hist, hist_naive, hist_sz);
  //printf("how fast?\n");
  compareTest<unsigned int>(hist, hist_naive, hist_sz);

  // printf("LARGE HISTOGRAM RESULTS\n");
  // printIntArraySeq(hist_naive, hist_sz);
  // printf("ASYNCHRONOUS HISTOGRAM RESULTS\n");
  // printIntArraySeq(hist, hist_sz);
  printf("passed : %s\n", result ? "true" : "false");
  //printf("LARGE SUM: %d, ASYNC SUM: %d\n", sumSeq(hist_naive, hist_sz), sumSeq(hist, hist_sz));
  

  cudaFree(d_data);
  free(data);
  free(out);
  free(out_async);
  free(hist);
}

int main(int argc, char **argv) {

  unsigned int image_sz, hist_sz;

  if (argc != 3) {
    image_sz = IMG_SIZE;
    hist_sz = HIST_SIZE;
  } else {
    sscanf(argv[1], "%u", &image_sz);
    sscanf(argv[2], "%u", &hist_sz);
    image_sz *= 8;
    hist_sz *= 8;
  }

  test_hist<float> (image_sz, hist_sz);
  
}
