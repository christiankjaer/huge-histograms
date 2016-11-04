#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "Kernels.cu.h"
#include "Host.cu.h"

#define IMG_SIZE 1024*1024*1024
#define HIST_SIZE 1024*400


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
void cpu_hist(size_t image_size, T* image, size_t hist_size, unsigned int* hist, T max_e) {
  unsigned int *inds = (unsigned int*) malloc(sizeof(unsigned int[image_size]));
  memset(hist, 0, sizeof(unsigned int)*hist_size);
  calc_indices<T>(image_size, hist_size, image, inds, max_e);
  fill_histogram(image_size, hist_size, inds, hist);
}



template <class T>
void test_hist(unsigned int image_sz, unsigned int hist_sz) {

  printf("\nTesting histogram with\nimage size: %d\nhist size:%d\n", image_sz, hist_sz);

  struct timeval t_start, t_end;
  unsigned long int elapsed;

  T *data = (T*) malloc(sizeof(T[image_sz]));
  unsigned int *hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));
  unsigned int *seq_hist = (unsigned int*) malloc(sizeof(unsigned int[hist_sz]));

  srand(time(NULL));

  for (size_t i = 0; i < image_sz; i++) {
    data[i] = ((T)rand()/(T)RAND_MAX) * 256.0;
  }

  gettimeofday(&t_start, NULL);
  cpu_hist<T>(image_sz, data, hist_sz, seq_hist, (T)256.0);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated (CPU) in %d ms\n", elapsed/1000);

  elapsed = hostStreamHistogram<T>(image_sz, data, hist_sz, hist, (T)256.0);
  printf("Histogram calculated (streaming) in %d ms\n", elapsed/1000);

  for (int i = 0; i < hist_sz; i++) {
    if (seq_hist[i] != hist[i]) {
      printf("INVALID (shared) %d != %d at %d\n", seq_hist[i], hist[i], i);
      break;
    }
  }
  free(data);
  free(hist);
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

  // test_hist<float>(image_sz, hist_sz);
  test_hist<float>(image_sz, hist_sz);
}
