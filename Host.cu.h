#ifndef HOST_HIST
#define HOST_HIST

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cub/cub.cuh>
#include "sequential/arraylib.cu.h"

#include "Kernels.cu.h"
#include "setup.cu.h"

#define NUM_STREAMS 1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

long int timeval_subtract(struct timeval* t2, struct timeval* t1) {
  long int diff = (t2->tv_sec - t1->tv_sec) * 1000000;
  diff += t2->tv_usec - t1->tv_usec;
  return diff;
}


// @summary : computes the maximum value in an array of values of type T.
template<class T>
T maximumElement(T* d_in, int arr_size){

  // Determine temporary device storage requirements
  T* d_max;
  T  h_max;
  gpuErrchk( cudaMalloc((void**)&d_max, sizeof(T)) );
  void*    d_temp_storage     = NULL;
  size_t   temp_storage_bytes = 0;

  // Dummy call, to set temp_storage values.
  gpuErrchk( cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, arr_size) );
  gpuErrchk( cudaMalloc(&d_temp_storage, temp_storage_bytes) );

  // Let Cub handle the reduction
  gpuErrchk( cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, arr_size) );

  // Copy back the reduced element
  gpuErrchk( cudaMemcpy(&h_max, d_max, sizeof(T), cudaMemcpyDeviceToHost) );

  // Clean up memory
  cudaFree(d_temp_storage);
  cudaFree(d_max);
  return h_max;
}


// @summary : Normalizes the dateset, and computes histogram indexes.
// @remarks : Asserts the value and index arrays to have the same sizes.
// @params  : block_size -> size of the CUDA blocks to be used
//            arr_size   -> size of the two arrays
//            boundary   -> the maximum element size
//            vals_h     -> a pointer to the host allocated values
//            inds_h     -> a pointer to the host allocated index array
template<class T>
void histVals2IndexDevice (unsigned int    arr_size,
                           T*                vals_d,
                           unsigned int      hist_size,
                           unsigned int*     inds_d){

  int num_blocks = ceil((float)arr_size / CUDA_BLOCK_SIZE);
  T   boundary   = maximumElement<T>(vals_d, arr_size);

  // Run indexing kernel
  histVals2IndexKernel<T><<<num_blocks, CUDA_BLOCK_SIZE>>>
    (vals_d, inds_d, arr_size, hist_size, boundary);
}

template<class T>
void histVals2IndexStream (unsigned int    arr_size,
                           T*                vals_d,
                           unsigned int      hist_size,
                           unsigned int*     inds_d,
                           T                 max_e,
                           cudaStream_t      stream=0){


  int num_blocks = ceil((float)arr_size / CUDA_BLOCK_SIZE);

  // Run indexing kernel
  histVals2IndexKernel<T><<<num_blocks, CUDA_BLOCK_SIZE, 0, stream>>>
    (vals_d, inds_d, arr_size, hist_size, max_e);
}


// @summary : partially sortes an index array in chunk size bounded segments
void radixSortDevice(unsigned int* unsorted_d,
                     unsigned int* sorted_d,
                     unsigned int  array_length,
                     unsigned int  hist_size){


  void   *d_temp_storage    = NULL;
  size_t temp_storage_bytes = 0;


  // TODO : Figure out an intelligent way of computing end bit [-.-]
  int begin_bit = ceil(log2((float) GPU_HIST_SIZE));
  int end_bit   = max(begin_bit, (int)ceil(log2((float) hist_size)))+1;
  // int end_bit = 8*sizeof(unsigned int);

  // Figure out how much temporary storage is needed
  gpuErrchk( cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 unsorted_d,
                                 sorted_d,
                                 array_length,
                                 begin_bit,
                                 end_bit));
  gpuErrchk( cudaMalloc(&d_temp_storage, temp_storage_bytes) );

  // Call the library function
  gpuErrchk (cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                 temp_storage_bytes,
                                 unsorted_d,
                                 sorted_d,
                                 array_length,
                                 begin_bit,
                                 end_bit) );

  cudaFree(d_temp_storage);
}


// @summary : Computes segment offsets for each
//            segment into the index array
void segmentOffsets(unsigned int  inds_size,
                    unsigned int* inds_d,
                    unsigned int* segment_sizes_d){

  int num_blocks = ceil((float)inds_size / CUDA_BLOCK_SIZE);

  segmentOffsetsKernel<<<num_blocks, CUDA_BLOCK_SIZE>>> (inds_d, inds_size, segment_sizes_d);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

// @summary : Host wrapper for the segmented histogram
//            kernel.
template <class T>
unsigned long int largeHistogram(unsigned int image_size,
                                 T* d_image,
                                 unsigned int histogram_size,
                                 unsigned int* d_hist) {


  unsigned int chunk_size = ceil((float)image_size / HARDWARE_PARALLELISM);
  unsigned int block_workload = chunk_size * CUDA_BLOCK_SIZE;
  unsigned int num_blocks = ceil((float)image_size / block_workload);
  unsigned int num_segments = ceil((float)histogram_size / GPU_HIST_SIZE);

  unsigned int *d_inds, *d_sorted, *d_sgm_offset;

  gpuErrchk( cudaMalloc(&d_inds, sizeof(unsigned int)*image_size) );
  gpuErrchk( cudaMalloc(&d_sorted, sizeof(unsigned int)*image_size) );
  gpuErrchk( cudaMalloc(&d_sgm_offset, sizeof(unsigned int)*num_segments) );

  struct timeval t_start, t_end;
  unsigned long int elapsed;

  gettimeofday(&t_start, NULL);

  histVals2IndexDevice<T>(image_size, d_image, histogram_size, d_inds);
  radixSortDevice(d_inds, d_sorted, image_size, histogram_size);
  segmentOffsets(image_size, d_sorted, d_sgm_offset);

  gpuErrchk( cudaMemset(d_hist, 0, sizeof(unsigned int)*histogram_size) );

  histKernel<<<num_blocks, CUDA_BLOCK_SIZE>>>
    (image_size, histogram_size, chunk_size, d_sgm_offset, d_sorted, d_hist);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);

  cudaFree(d_inds);
  cudaFree(d_sgm_offset);
  return elapsed;
}

// @summary : Host wrapper for the segmented histogram
//            kernel.
template <class T>
unsigned long int smallHistogram(unsigned int image_size,
                                 T* d_image,
                                 unsigned int histogram_size,
                                 unsigned int* d_hist) {


  unsigned int chunk_size = ceil((float)image_size / HARDWARE_PARALLELISM);
  unsigned int block_workload = chunk_size * CUDA_BLOCK_SIZE;
  unsigned int num_blocks = ceil((float)image_size / block_workload);

  unsigned int *d_inds;

  gpuErrchk( cudaMalloc(&d_inds, sizeof(unsigned int)*image_size) );

  struct timeval t_start, t_end;
  unsigned long int elapsed;

  gettimeofday(&t_start, NULL);

  histVals2IndexDevice<T>(image_size, d_image, histogram_size, d_inds);

  gpuErrchk( cudaMemset(d_hist, 0, sizeof(unsigned int)*histogram_size) );

  smallHistKernel<<<num_blocks, CUDA_BLOCK_SIZE>>>
    (image_size, d_inds, chunk_size, histogram_size, d_hist);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);

  cudaFree(d_inds);
  return elapsed;
}

template <class T>
unsigned long int naiveHistogram(unsigned int  image_size,
                                 T*            d_image,
                                 unsigned int  histogram_size,
                                 unsigned int* d_hist) {


  unsigned int num_blocks = ceil((float)image_size / CUDA_BLOCK_SIZE);

  unsigned int *d_inds;

  gpuErrchk( cudaMalloc(&d_inds, sizeof(unsigned int)*image_size) );

  struct timeval t_start, t_end;
  unsigned long int elapsed;
  gettimeofday(&t_start, NULL);

  histVals2IndexDevice<T>(image_size, d_image, histogram_size, d_inds);

  gpuErrchk( cudaMemset(d_hist, 0, sizeof(unsigned int)*histogram_size) );

  naiveHistKernel<<<num_blocks, CUDA_BLOCK_SIZE>>>(image_size, d_inds, d_hist);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);

  cudaFree(d_inds);
  return elapsed;
}

template <class T>
void streamHistogram(unsigned int image_size,
                     T* d_image,
                     unsigned int *d_inds,
                     unsigned int histogram_size,
                     unsigned int* d_hist,
                     T max_e,
                     cudaStream_t stream=0) {


  unsigned int num_blocks = ceil((float)image_size / CUDA_BLOCK_SIZE);

  histVals2IndexStream<T>(image_size, d_image, histogram_size, d_inds, max_e, stream);

  naiveHistKernel<<<num_blocks, CUDA_BLOCK_SIZE, 0, stream>>>(image_size, d_inds, d_hist);
}

template <class T>
unsigned int hostStreamHistogram(unsigned int  image_size,
                                 T*            image,
                                 unsigned int  hist_size,
                                 unsigned int* hist,
                                 T             max_e) {


  unsigned int buf_size = 1024*1024;

  T *d_img_buf[NUM_STREAMS];
  unsigned int *d_inds_buf[NUM_STREAMS], *d_hist;

  cudaMalloc(&d_hist, sizeof(unsigned int)*hist_size);

  cudaStream_t streams[NUM_STREAMS];

  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaMalloc(&d_img_buf[i], sizeof(T)*buf_size);
    cudaMalloc(&d_inds_buf[i], sizeof(unsigned int)*buf_size);
    gpuErrchk( cudaStreamCreate(&streams[i]) );
  }

  cudaMemset(d_hist, 0, sizeof(unsigned int)*hist_size);

  struct timeval t_start, t_end;
  unsigned long int elapsed;
  gettimeofday(&t_start, NULL);

  for (unsigned int i = 0, j = 0; i < image_size; i += buf_size, j++) {
    unsigned int size = min(i+buf_size, image_size) - i;

    cudaMemcpyAsync(d_img_buf[j%NUM_STREAMS], &image[i], size*sizeof(T),
                    cudaMemcpyHostToDevice, streams[j%NUM_STREAMS]);

    streamHistogram<T>(size, d_img_buf[j%NUM_STREAMS], d_inds_buf[j%NUM_STREAMS], hist_size,
                       d_hist, max_e, streams[j%NUM_STREAMS]);
  }

  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaMemcpy(hist, d_hist, sizeof(int)*hist_size, cudaMemcpyDeviceToHost);

  gpuErrchk( cudaDeviceSynchronize() );

  return elapsed;

}

#endif //HOST_HIST
