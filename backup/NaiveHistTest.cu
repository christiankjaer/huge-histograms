#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "Kernels.cu.h"

#define IMG_SIZE 8192*1024
#define HIST_SIZE 8192

long int timeval_subtract(struct timeval* t2, struct timeval* t1) {
  long int diff = (t2->tv_sec - t1->tv_sec) * 1000000;
  diff += t2->tv_usec - t1->tv_usec;
  return diff;
}

void calc_indices(size_t N, size_t M, float *in, int *out, float boundary) {
  for (size_t i = 0; i < N; i++) {
    out[i] = (int) ((in[i] / boundary) * (float) M);
  }
}

void fill_histogram(size_t N, size_t M,
                    int *in, int *hist) {
  for (size_t i = 0; i < N; i++) {
    hist[in[i]]++;
  }
}

int main() {
  struct timeval t_start, t_end;
  unsigned long int elapsed;

  float *data = (float*) malloc(sizeof(float[IMG_SIZE]));
  int *inds = (int*) malloc(sizeof(size_t[IMG_SIZE]));
  int *hist = (int*) malloc(sizeof(unsigned int[HIST_SIZE]));

  int *seq_hist = (int*) malloc(sizeof(unsigned int[HIST_SIZE]));
  memset(seq_hist, 0, sizeof(unsigned int)*HIST_SIZE);

  int *d_inds, *d_hist;

  cudaMalloc(&d_inds, sizeof(int)*IMG_SIZE);
  cudaMalloc(&d_hist, sizeof(int)*HIST_SIZE);

  for (size_t i = 0; i < IMG_SIZE; i++) {
    data[i] = ((float)rand()/(float)RAND_MAX) * 256.0f;
  }
  calc_indices(IMG_SIZE, HIST_SIZE, data, inds, 256.0f);

  gettimeofday(&t_start, NULL);
  fill_histogram(IMG_SIZE, HIST_SIZE, inds, seq_hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated in %d µs\n", elapsed);

  cudaMemcpy(d_inds, inds, sizeof(int)*IMG_SIZE, cudaMemcpyHostToDevice);

  gettimeofday(&t_start, NULL);

  naiveHistKernel<<<1, 1024>>>(IMG_SIZE, d_inds, d_hist);
  cudaThreadSynchronize();

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated in %d µs\n", elapsed);

  cudaMemcpy(hist, d_hist, sizeof(int)*HIST_SIZE, cudaMemcpyDeviceToHost);

  for (int i = 0; i < HIST_SIZE; i++) {
    if (seq_hist[i] != hist[i]) {
      printf("INVALID %d != %d\n", seq_hist[i], hist[i]);
      break;
    }
  }
}