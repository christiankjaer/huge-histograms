#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

#define IMG_SIZE 8192*8192
#define HIST_SIZE 32

long int timeval_subtract(struct timeval* t2, struct timeval* t1) {
  long int diff = (t2->tv_sec - t1->tv_sec) * 1000000;
  diff += t2->tv_usec - t1->tv_usec;
  return diff;
}

void calc_indices(size_t N, size_t M, float in[N], size_t out[N], float boundary) {
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    out[i] = (size_t) ((in[i] / boundary) * (float) M);
  }
}

void fill_histogram(size_t N, size_t M,
                    size_t in[N], unsigned int hist[M]) {
  for (size_t i = 0; i < N; i++) {
    hist[in[i]]++;
  }
}

int main(int argc, char **argv) {

  struct timeval t_start, t_end;
  unsigned long int elapsed;
  srand(time(NULL));

  float *data = malloc(sizeof(float[IMG_SIZE]));
  size_t *inds = malloc(sizeof(size_t[IMG_SIZE]));
  unsigned int *hist = malloc(sizeof(unsigned int[HIST_SIZE]));

  memset(hist, 0, sizeof(unsigned int)*HIST_SIZE);

  for (size_t i = 0; i < IMG_SIZE; i++) {
    data[i] = ((float)rand()/(float)RAND_MAX) * 256.0f;
  }
  gettimeofday(&t_start, NULL);
  calc_indices(IMG_SIZE, HIST_SIZE, data, inds, 256.0f);
  fill_histogram(IMG_SIZE, HIST_SIZE, inds, hist);
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Histogram calculated in %d µs\n", elapsed);

  free(data); free(inds); free(hist);
}
