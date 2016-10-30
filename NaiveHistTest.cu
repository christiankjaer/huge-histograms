#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "Kernels.cu.h"
#include "Host.cu.h"

#define IMG_SIZE 8192*768
#define HIST_SIZE 8192*16

long int timeval_subtract(struct timeval* t2, struct timeval* t1) {
  long int diff = (t2->tv_sec - t1->tv_sec) * 1000000;
  diff += t2->tv_usec - t1->tv_usec;
  return diff;
}

void calc_indices(size_t N, size_t M, float *in, unsigned int *out, float boundary) {
  for (size_t i = 0; i < N; i++) {
    out[i] = (unsigned int) ((in[i] / boundary) * (float) M);
  }
}

void fill_histogram(size_t N, size_t M,
                    unsigned int *in, unsigned int *hist) {
  for (size_t i = 0; i < N; i++) {
    hist[in[i]]++;
  }
}

float max_elem(size_t arr_size, float* arr) {
  if (arr_size == 0) {
    return 0.0;
  }
  float maxe = arr[0];
  for (int i = 1; i < arr_size; i++) {
    maxe = arr[i] > maxe ? arr[i] : maxe;
  }
  return maxe;
}

void cpu_hist(size_t image_size, float* image, size_t hist_size, unsigned int* hist) {
  unsigned int *inds = (unsigned int*) malloc(sizeof(unsigned int[image_size]));
  memset(hist, 0, sizeof(unsigned int)*hist_size);
  calc_indices(image_size, hist_size, image, inds, max_elem(image_size, image));
  fill_histogram(image_size, HIST_SIZE, inds, hist);
}

int main() {

  unsigned int chunk_size = ceil((float)IMG_SIZE / HARDWARE_PARALLELISM);
  unsigned int block_workload = chunk_size * CUDA_BLOCK_SIZE;
  unsigned int num_blocks = ceil((float)IMG_SIZE / block_workload);
  unsigned int num_segments = ceil((float)HIST_SIZE / GPU_HIST_SIZE);
  
  struct timeval t_start, t_end;
  unsigned long int elapsed;

  float *data = (float*) malloc(sizeof(float[IMG_SIZE]));
  unsigned int *hist = (unsigned int*) malloc(sizeof(unsigned int[HIST_SIZE]));
  unsigned int *seq_hist = (unsigned int*) malloc(sizeof(unsigned int[HIST_SIZE]));


  float *d_image;
  unsigned int *d_hist;

  cudaMalloc(&d_image, sizeof(float)*IMG_SIZE);
  cudaMalloc(&d_hist, sizeof(int)*HIST_SIZE);

  for (size_t i = 0; i < IMG_SIZE; i++) {
    data[i] = ((float)rand()/(float)RAND_MAX) * 256.0f;
  }

  gettimeofday(&t_start, NULL);
  cpu_hist(IMG_SIZE, data, HIST_SIZE, seq_hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated in %d µs\n", elapsed);

  cudaMemcpy(d_image, data, sizeof(int)*IMG_SIZE, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();

  gettimeofday(&t_start, NULL);

  largeHistogram<float>(IMG_SIZE, d_image, HIST_SIZE, d_hist);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated in %d µs\n", elapsed);

  cudaMemcpy(hist, d_hist, sizeof(int)*HIST_SIZE, cudaMemcpyDeviceToHost);

  for (int i = 0; i < HIST_SIZE; i++) {
    if (seq_hist[i] != hist[i]) {
      printf("INVALID %d != %d\n at %d", seq_hist[i], hist[i], i);
      break;
    }
  }
}
