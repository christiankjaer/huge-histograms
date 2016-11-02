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

template <class T>
void cpu_hist(size_t image_size, T* image, size_t hist_size, unsigned int* hist) {
  unsigned int *inds = (unsigned int*) malloc(sizeof(unsigned int[image_size]));
  memset(hist, 0, sizeof(unsigned int)*hist_size);
  calc_indices<T>(image_size, hist_size, image, inds, max_elem<T>(image_size, image));
  fill_histogram(image_size, hist_size, inds, hist);
}

template <class T>
void bench_hist(unsigned int image_sz, unsigned int hist_sz) {

  struct timeval t_start, t_end;
  unsigned long int elapsed;

  T *data = (T*) malloc(sizeof(T[image_sz]));
  unsigned int *seq_hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));

  T *d_image;
  unsigned int *d_hist, *d_hist2;

  gpuErrchk( cudaMalloc(&d_image, sizeof(T)*image_sz) );
  gpuErrchk( cudaMalloc(&d_hist, sizeof(int)*hist_sz) );
  gpuErrchk( cudaMalloc(&d_hist2, sizeof(int)*hist_sz) );

  srand(time(NULL));

  for (size_t i = 0; i < image_sz; i++) {
    data[i] = ((T)rand()/(T)RAND_MAX) * 256.0;
  }

  gettimeofday(&t_start, NULL);
  cpu_hist<T>(image_sz, data, hist_sz, seq_hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("%d %d %d ", image_sz, hist_sz, elapsed);

  gpuErrchk( cudaMemcpy(d_image, data, sizeof(T)*image_sz, cudaMemcpyHostToDevice) );

  elapsed = naiveHistogram<T>(image_sz, d_image, hist_sz, d_hist2);
  printf("%d ", elapsed);

  elapsed = largeHistogram<T>(image_sz, d_image, hist_sz, d_hist);
  printf("%d\n", elapsed);

  cudaFree(d_image);
  cudaFree(d_hist);
  cudaFree(d_hist2);
  free(data);
  free(seq_hist);
}

template <class T>
void bench_small_hist(unsigned int image_sz, unsigned int hist_sz) {

  struct timeval t_start, t_end;
  unsigned long int elapsed;

  T *data = (T*) malloc(sizeof(T[image_sz]));
  unsigned int *seq_hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));

  T *d_image;
  unsigned int *d_hist, *d_hist2;

  gpuErrchk( cudaMalloc(&d_image, sizeof(T)*image_sz) );
  gpuErrchk( cudaMalloc(&d_hist, sizeof(int)*hist_sz) );
  gpuErrchk( cudaMalloc(&d_hist2, sizeof(int)*hist_sz) );

  srand(time(NULL));

  for (size_t i = 0; i < image_sz; i++) {
    data[i] = ((T)rand()/(T)RAND_MAX) * 256.0;
  }

  gettimeofday(&t_start, NULL);
  cpu_hist<T>(image_sz, data, hist_sz, seq_hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("%d %d %d ", image_sz, hist_sz, elapsed);

  gpuErrchk( cudaMemcpy(d_image, data, sizeof(T)*image_sz, cudaMemcpyHostToDevice) );

  elapsed = naiveHistogram<T>(image_sz, d_image, hist_sz, d_hist2);
  printf("%d ", elapsed);

  elapsed = smallHistogram<T>(image_sz, d_image, hist_sz, d_hist);
  printf("%d\n", elapsed);

  cudaFree(d_image);
  cudaFree(d_hist);
  cudaFree(d_hist2);
  free(data);
  free(seq_hist);
}

template <class T>
void test_small_hist(unsigned int image_sz, unsigned int hist_sz) {

  printf("\nTesting histogram with\nimage size: %d\nhist size:%d\n", image_sz, hist_sz);
  
  struct timeval t_start, t_end;
  unsigned long int elapsed;

  T *data = (T*) malloc(sizeof(T[image_sz]));
  unsigned int *hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));
  unsigned int *hist1 = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));
  unsigned int *hist2 = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));
  unsigned int *seq_hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));


  T *d_image;
  unsigned int *d_hist, *d_hist1, *d_hist2;

  gpuErrchk( cudaMalloc(&d_image, sizeof(T)*image_sz) );
  gpuErrchk( cudaMalloc(&d_hist, sizeof(int)*hist_sz) );
  gpuErrchk( cudaMalloc(&d_hist1, sizeof(int)*hist_sz) );
  gpuErrchk( cudaMalloc(&d_hist2, sizeof(int)*hist_sz) );

  srand(time(NULL));

  for (size_t i = 0; i < image_sz; i++) {
    data[i] = ((T)rand()/(T)RAND_MAX) * 256.0;
  }

  gettimeofday(&t_start, NULL);
  cpu_hist<T>(image_sz, data, hist_sz, seq_hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated (CPU) in %d µs\n", elapsed);

  gpuErrchk( cudaMemcpy(d_image, data, sizeof(T)*image_sz, cudaMemcpyHostToDevice) );

  elapsed = naiveHistogram<T>(image_sz, d_image, hist_sz, d_hist2);
  printf("Histogram calculated (GPU naive) in %d µs\n", elapsed);

  elapsed = largeHistogram<T>(image_sz, d_image, hist_sz, d_hist);
  printf("Histogram calculated (GPU shared) in %d µs\n", elapsed);

  elapsed = smallHistogram<T>(image_sz, d_image, hist_sz, d_hist1);
  printf("Histogram calculated (GPU small) in %d µs\n", elapsed);


  gpuErrchk( cudaMemcpy(hist, d_hist, sizeof(int)*hist_sz, cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(hist1, d_hist1, sizeof(int)*hist_sz, cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(hist2, d_hist2, sizeof(int)*hist_sz, cudaMemcpyDeviceToHost) );

  for (int i = 0; i < hist_sz; i++) {
    if (seq_hist[i] != hist[i]) {
      printf("INVALID (shared) %d != %d at %d\n", seq_hist[i], hist[i], i);
      break;
    }
    if (seq_hist[i] != hist2[i]) {
      printf("INVALID (naive) %d != %d at %d\n", seq_hist[i], hist2[i], i);
      break;
    }
    if (seq_hist[i] != hist1[i]) {
      printf("INVALID (small) %d != %d at %d\n", seq_hist[i], hist1[i], i);
      break;
    }
  }
  cudaFree(d_image);
  cudaFree(d_hist);
  cudaFree(d_hist1);
  cudaFree(d_hist2);
  free(data);
  free(hist);
  free(hist1);
  free(hist2);
  free(seq_hist);
}

template <class T>
void test_hist(unsigned int image_sz, unsigned int hist_sz) {

  printf("\nTesting histogram with\nimage size: %d\nhist size:%d\n", image_sz, hist_sz);

  struct timeval t_start, t_end;
  unsigned long int elapsed;

  T *data = (T*) malloc(sizeof(T[image_sz]));
  unsigned int *hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));
  unsigned int *hist2 = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));
  unsigned int *seq_hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));


  T *d_image;
  unsigned int *d_hist, *d_hist2;

  gpuErrchk( cudaMalloc(&d_image, sizeof(T)*image_sz) );
  gpuErrchk( cudaMalloc(&d_hist, sizeof(int)*hist_sz) );
  gpuErrchk( cudaMalloc(&d_hist2, sizeof(int)*hist_sz) );

  srand(time(NULL));

  for (size_t i = 0; i < image_sz; i++) {
    data[i] = ((T)rand()/(T)RAND_MAX) * 256.0;
  }

  gettimeofday(&t_start, NULL);
  cpu_hist<T>(image_sz, data, hist_sz, seq_hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated (CPU) in %d µs\n", elapsed);

  gpuErrchk( cudaMemcpy(d_image, data, sizeof(T)*image_sz, cudaMemcpyHostToDevice) );

  elapsed = largeHistogram<T>(image_sz, d_image, hist_sz, d_hist);
  printf("Histogram calculated (GPU shared) in %d µs\n", elapsed);

  elapsed = naiveHistogram<T>(image_sz, d_image, hist_sz, d_hist2);
  printf("Histogram calculated (GPU naive) in %d µs\n", elapsed);

  gpuErrchk( cudaMemcpy(hist, d_hist, sizeof(int)*hist_sz, cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(hist2, d_hist2, sizeof(int)*hist_sz, cudaMemcpyDeviceToHost) );

  for (int i = 0; i < hist_sz; i++) {
    if (seq_hist[i] != hist[i]) {
      printf("INVALID (shared) %d != %d at %d\n", seq_hist[i], hist[i], i);
      break;
    }
    if (seq_hist[i] != hist2[i]) {
      printf("INVALID (naive) %d != %d at %d\n", seq_hist[i], hist2[i], i);
      break;
    }
  }
  cudaFree(d_image);
  cudaFree(d_hist);
  cudaFree(d_hist2);
  free(data);
  free(hist);
  free(hist2);
  free(seq_hist);
}

int main(int argc, char **argv) {

  unsigned int image_sz, hist_sz;

  if (argc != 3) {
    image_sz = IMG_SIZE;
    hist_sz = HIST_SIZE;
  } else {
    sscanf(argv[1], "%u", &image_sz);
    sscanf(argv[2], "%u", &hist_sz);
  }

  //test_small_hist<float>(image_sz, hist_sz);
  bench_small_hist<float>(image_sz, hist_sz);
}
