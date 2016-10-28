#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "Kernels.cu.h"
#include "Host.cu.h"

#define IMG_SIZE 8192*768
#define HIST_SIZE 8192

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

int main() {

  unsigned int chunk_size = ceil((float)IMG_SIZE / HARDWARE_PARALLELISM);
  unsigned int block_workload = chunk_size * CUDA_BLOCK_SIZE;
  unsigned int num_blocks = ceil((float)IMG_SIZE / block_workload);
  unsigned int num_segments = ceil((float)HIST_SIZE / GPU_HIST_SIZE);
  
  struct timeval t_start, t_end;
  unsigned long int elapsed;

  float *data = (float*) malloc(sizeof(float[IMG_SIZE]));
  unsigned int *inds = (unsigned int*) malloc(sizeof(unsigned int[IMG_SIZE]));
  unsigned int *hist = (unsigned int*) malloc(sizeof(unsigned int[HIST_SIZE]));
  unsigned int *seq_hist = (unsigned int*) malloc(sizeof(unsigned int[HIST_SIZE]));

  memset(seq_hist, 0, sizeof(unsigned int)*HIST_SIZE);

  unsigned int *d_inds, *d_hist, *d_sgm_idx, *d_sgm_offset;
  unsigned int *sgm_idx = (unsigned int*) malloc(sizeof(unsigned int)*num_blocks);
  unsigned int *sgm_offset = (unsigned int*) malloc(sizeof(unsigned int)*num_segments);

  cudaMalloc(&d_inds, sizeof(int)*IMG_SIZE);
  cudaMalloc(&d_hist, sizeof(int)*HIST_SIZE);
  cudaMalloc(&d_sgm_idx, sizeof(int)*num_blocks);
  cudaMalloc(&d_sgm_offset, sizeof(int)*num_segments);

  for (size_t i = 0; i < IMG_SIZE; i++) {
    data[i] = ((float)rand()/(float)RAND_MAX) * 256.0f;
  }
  calc_indices(IMG_SIZE, HIST_SIZE, data, inds, 256.0f);

  gettimeofday(&t_start, NULL);
  fill_histogram(IMG_SIZE, HIST_SIZE, inds, seq_hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated in %d µs\n", elapsed);

  radixSort(inds, IMG_SIZE);
  cudaMemcpy(d_inds, inds, sizeof(int)*IMG_SIZE, cudaMemcpyHostToDevice);

  metaData(IMG_SIZE, d_inds, num_segments, block_workload, d_sgm_offset, d_sgm_idx);
  cudaMemcpy(sgm_idx, d_sgm_idx, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost);
  cudaMemcpy(sgm_offset, d_sgm_offset, sizeof(int)*num_segments, cudaMemcpyDeviceToHost);


  // printf("num_sgm: %d\n", num_segments);
  // printf("num_blocks: %d\n", num_blocks);
  // printIntArraySeq<unsigned int>(inds, IMG_SIZE);
  // printIntArraySeq<unsigned int>(sgm_idx, num_blocks);
  // printIntArraySeq<unsigned int>(sgm_offset, num_segments);

  gettimeofday(&t_start, NULL);

  cudaMemset(d_hist, 0, sizeof(int)*HIST_SIZE);

  christiansHistKernel<<<num_blocks, CUDA_BLOCK_SIZE>>>
    (IMG_SIZE, HIST_SIZE, chunk_size, d_sgm_idx, d_sgm_offset, d_inds, d_hist);
  cudaThreadSynchronize();
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
