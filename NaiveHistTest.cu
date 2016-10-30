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

long int timeval_subtract(struct timeval* t2, struct timeval* t1) {
  long int diff = (t2->tv_sec - t1->tv_sec) * 1000000;
  diff += t2->tv_usec - t1->tv_usec;
  return diff;
}

void calc_indices(size_t N, size_t M, float *in, unsigned int *out, float boundary) {
  for (size_t i = 0; i < N; i++) {
    out[i] = (unsigned int) ((in[i] / boundary) * (float) (M - 1));
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
  fill_histogram(image_size, hist_size, inds, hist);
}

void test_sgm_offset(unsigned int image_sz, unsigned int hist_sz) {

  unsigned int chunk_size = ceil((float)image_sz / HARDWARE_PARALLELISM);
  unsigned int block_workload = chunk_size * CUDA_BLOCK_SIZE;
  unsigned int num_blocks = ceil((float)image_sz / block_workload);
  unsigned int num_segments = ceil((float)hist_sz / GPU_HIST_SIZE);

  float *data = (float*) malloc(sizeof(float[image_sz]));
  unsigned int *h_sort = (unsigned int*) malloc(sizeof(unsigned int)*image_sz);
  unsigned int *h_sgm_idx = (unsigned int*) malloc(sizeof(unsigned int)*num_blocks);
  unsigned int *h_sgm_offset = (unsigned int*) malloc(sizeof(unsigned int)*num_segments);

  float *d_image;
  unsigned int *d_inds, *d_sgm_idx, *d_sgm_offset, *d_sorted;

  cudaMalloc(&d_inds, sizeof(unsigned int)*image_sz);
  cudaMalloc(&d_sorted, sizeof(unsigned int)*image_sz);
  cudaMalloc(&d_sgm_idx, sizeof(unsigned int)*num_blocks);
  cudaMalloc(&d_sgm_offset, sizeof(unsigned int)*num_segments);
  cudaMalloc(&d_image, sizeof(float)*image_sz);

  srand(time(NULL));

  for (size_t i = 0; i < image_sz; i++) {
    data[i] = ((float)rand()/(float)RAND_MAX) * 256.0f;
  }

  cudaMemcpy(d_image, data, sizeof(int)*image_sz, cudaMemcpyHostToDevice);

  histVals2IndexDevice<float>(image_sz, d_image, hist_sz, d_inds);
  radixSortDevice(d_inds, d_sorted, image_sz);
  metaData(image_sz, d_sorted, num_segments, block_workload, d_sgm_offset, d_sgm_idx);

  cudaMemcpy(h_sgm_offset, d_sgm_offset, sizeof(unsigned int)*num_segments, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sgm_idx, d_sgm_idx, sizeof(unsigned int)*num_blocks, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sort, d_sorted, sizeof(unsigned int)*image_sz, cudaMemcpyDeviceToHost);

  cudaThreadSynchronize();

  printIntArraySeq<unsigned int>(h_sort, image_sz);
  printIntArraySeq<unsigned int>(h_sgm_offset, num_segments);
  printIntArraySeq<unsigned int>(h_sgm_idx, num_blocks);

}

void test_hist(unsigned int image_sz, unsigned int hist_sz) {

  unsigned int chunk_size = ceil((float)image_sz / HARDWARE_PARALLELISM);
  unsigned int block_workload = chunk_size * CUDA_BLOCK_SIZE;
  unsigned int num_blocks = ceil((float)image_sz / block_workload);
  unsigned int num_segments = ceil((float)hist_sz / GPU_HIST_SIZE);
  
  struct timeval t_start, t_end;
  unsigned long int elapsed;

  float *data = (float*) malloc(sizeof(float[image_sz]));
  unsigned int *hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));
  unsigned int *seq_hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));


  float *d_image;
  unsigned int *d_hist;

  cudaMalloc(&d_image, sizeof(float)*image_sz);
  cudaMalloc(&d_hist, sizeof(int)*hist_sz);

  srand(time(NULL));

  for (size_t i = 0; i < image_sz; i++) {
    data[i] = ((float)rand()/(float)RAND_MAX) * 256.0f;
  }

  gettimeofday(&t_start, NULL);
  cpu_hist(image_sz, data, hist_sz, seq_hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated in %d µs\n", elapsed);

  cudaMemcpy(d_image, data, sizeof(int)*image_sz, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();

  gettimeofday(&t_start, NULL);

  largeHistogram<float>(image_sz, d_image, hist_sz, d_hist);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated in %d µs\n", elapsed);

  cudaMemcpy(hist, d_hist, sizeof(int)*hist_sz, cudaMemcpyDeviceToHost);

  for (int i = 0; i < hist_sz; i++) {
    if (seq_hist[i] != hist[i]) {
      printf("INVALID %d != %d at %d\n", seq_hist[i], hist[i], i);
      break;
    }
  }
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

  test_hist(image_sz, hist_sz);
  //test_sgm_offset(image_sz, hist_sz);
}
