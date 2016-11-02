#ifndef HOST_HIST
#define HOST_HIST

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

#include <cub/cub.cuh>
#include "sequential/arraylib.cu.h"

#include "Kernels.cu.h"
#include "setup.cu.h"
#include "struct.cu.h"

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

// @summary : computes the maximum value in an array of values of type T.
template<class T>
void maximumElementDevice(T* d_in, T* max, int arr_size){

  // Determine temporary device storage requirements
  T* d_max;
  //T  h_max;
  gpuErrchk( cudaMalloc((void**)&d_max, sizeof(T)) );
  void*    d_temp_storage     = NULL;
  size_t   temp_storage_bytes = 0;

  // Dummy call, to set temp_storage values.¨
  gpuErrchk( cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, arr_size) );
  gpuErrchk( cudaMalloc(&d_temp_storage, temp_storage_bytes) );

  // Let Cub handle the reduction
  gpuErrchk( cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, arr_size) );

  // Copy back the reduced element
  max[0] = d_max[0];
  //gpuErrchk( cudaMemcpy(&h_max, d_max, sizeof(T), cudaMemcpyDeviceToHost) );

  // Clean up memory
  cudaFree(d_temp_storage);
  cudaFree(d_max);
  //return d_max;
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

// @summary : Normalizes the dateset, and computes histogram indexes.
// @remarks : Asserts the value and index arrays to have the same sizes.
// @params  : block_size -> size of the CUDA blocks to be used
//            arr_size   -> size of the two arrays
//            boundary   -> the maximum element size
//            vals_h     -> a pointer to the host allocated values
//            inds_h     -> a pointer to the host allocated index array
//            Boundary   -> boundary element to split bin intervals
template<class T>
void histVals2IndexDeviceMax (unsigned int    arr_size,
                              T*                vals_d,
                              unsigned int      hist_size,
                              unsigned int*     inds_d,
                              T               boundary){

  int num_blocks = ceil((float)arr_size / CUDA_BLOCK_SIZE);
  //T   boundary   = maximumElement<T>(vals_d, arr_size);

  // Run indexing kernel
  histVals2IndexKernel<T><<<num_blocks, CUDA_BLOCK_SIZE>>>
    (vals_d, inds_d, arr_size, hist_size, boundary);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
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


  unsigned int num_blocks = ceil((float)image_size / CUDA_BLOCK_SIZE);

  unsigned int *d_inds;

  gpuErrchk( cudaMalloc(&d_inds, sizeof(unsigned int)*image_size) );

  struct timeval t_start, t_end;
  unsigned long int elapsed;

  gettimeofday(&t_start, NULL);

  histVals2IndexDevice<T>(image_size, d_image, histogram_size, d_inds);

  gpuErrchk( cudaMemset(d_hist, 0, sizeof(unsigned int)*histogram_size) );

  smallHistKernel<<<num_blocks, CUDA_BLOCK_SIZE>>>
    (image_size, d_inds, histogram_size, d_hist);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);

  cudaFree(d_inds);
  return elapsed;
}

template <class T>
unsigned long int naiveHistogram(unsigned int image_size,
                                 T* d_image,
                                 unsigned int histogram_size,
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
void naiveSquare(unsigned int tot_size,
                 T*               h_in,
                 T*              h_out){

  unsigned int num_blocks = (unsigned int) ceil(((float)tot_size)/ ((float)CUDA_BLOCK_SIZE));  
  // Initializes device arrays
  T* d_in, *d_out;
  gpuErrchk( cudaMalloc(&d_in, sizeof(T)*tot_size) );
  gpuErrchk( cudaMalloc(&d_out, sizeof(T)*tot_size) );
  gpuErrchk( cudaMemset(d_out, 0, sizeof(T)*tot_size) );

  struct timeval t_start, t_end;
  unsigned long int elapsed;
  gettimeofday(&t_start, NULL);
  
  gpuErrchk( cudaMemcpy(d_in, h_in, tot_size*sizeof(T), cudaMemcpyHostToDevice) );

  // executes kernels
  shitKernel<<<num_blocks,CUDA_BLOCK_SIZE>>>
    (d_in, d_out, tot_size);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // copies memory back
  gpuErrchk( cudaMemcpy(h_out, d_out, tot_size*sizeof(T), cudaMemcpyDeviceToHost) );
  
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("square naive (GPU) in %d µs\n", elapsed);


  // free device memory
  cudaFree(d_in);
  cudaFree(d_out);
}


// @summary : Takes an array and returns and array of squared elements
// @remarks : Just give it a large input, and it will compute square of each element
// @params  : array_size     -> size of image
//          : h_in           -> input array
//          : h_out          -> for all h_out[i] = h_in[i]^2

template <class T>
void asyncSquare(unsigned int array_size,
                 T*                 h_in,
                 T*                h_out){

  unsigned int stream_entries = 2;
  unsigned int stream_size = MAXIMUM_STREAM_SIZE*stream_entries;
  unsigned int nStreams = 0;

  while (nStreams < stream_entries) {
    nStreams = ceil((float)array_size / stream_size);
    stream_size = stream_size/2;
  }
  
  unsigned int num_blocks = ceil((float)stream_size/CUDA_BLOCK_SIZE);

  printf("number of streams : %d, stream_size : %d\n", nStreams, stream_size);
  // Initializes device arrays
  T* d_in, *d_out;
  cudaMalloc((void**) &d_in, stream_size*stream_entries*sizeof(T));
  cudaMalloc((void**) &d_out, stream_size*stream_entries*sizeof(T));

  // Initailizes streams
  cudaStream_t stream[stream_entries];
  for (int i = 0; i < stream_entries; i++){
    gpuErrchk( cudaStreamCreate(&stream[i]) );
  }

  struct timeval t_start, t_end;
  unsigned long int elapsed;
  gettimeofday(&t_start, NULL);
  for (int k = 0; k < nStreams; k++){
    // copies to asynchronously to the streams
    for (int i = 0; i < stream_entries; ++i) {
      int offset = i * stream_size;
      gpuErrchk( cudaMemcpyAsync(&d_in[offset], &h_in[k*offset],
                                   stream_size*sizeof(T), cudaMemcpyHostToDevice, stream[i]));
    }
    
    // executes kernels
    for (int i = 0; i < stream_entries; ++i) {
      int offset = i * STREAM_SIZE_GPU;
      shitKernel<<<num_blocks, CUDA_BLOCK_SIZE, 0, stream[i]>>>
        (&d_in[offset], &d_out[offset], offset);
      gpuErrchk( cudaPeekAtLastError() );
    }

    // asynchronous copy back from device to host
    for (int i = 0; i < stream_entries; ++i) {
      int offset = i * stream_size;
      gpuErrchk( cudaMemcpyAsync(&h_out[k*offset], &d_out[offset],
                                 stream_size*sizeof(T), cudaMemcpyDeviceToHost, stream[i]));
    }
  }
  
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("Square asynchronous (GPU) in %d µs\n", elapsed);

  // Destroys streams once done
  for(int i = 0; i < stream_entries; i++){
    gpuErrchk( cudaStreamDestroy(stream[i]) );
  }
  
  // free device memory
  cudaFree(d_in);
  cudaFree(d_out);
}

// @summary : Host wrapper for the segmented histogram
//            kernel.
template <class T>
unsigned long int largeHistogramAsync(unsigned int image_size,
                                      T* d_image,
                                      unsigned int *d_inds,
                                      unsigned int *d_sorted, 
                                      unsigned int *d_sgm_offset,
                                      unsigned int histogram_size,
                                      unsigned int* d_hist,
                                      T boundary,
                                      cudaStream_t stream) {

  unsigned int chunk_size = ceil((float)image_size / (HARDWARE_PARALLELISM));
  unsigned int block_workload = chunk_size * CUDA_BLOCK_SIZE;
  unsigned int num_blocks = ceil((float)image_size / block_workload);

  struct timeval t_start, t_end;
  unsigned long int elapsed;
  gettimeofday(&t_start, NULL);


  histVals2IndexDeviceMax<T>(image_size, d_image, histogram_size, d_inds, boundary);


  radixSortDevice(d_inds, d_sorted, image_size, histogram_size);

  segmentOffsets(image_size, d_sorted, d_sgm_offset);
  
  histKernel<<<num_blocks, CUDA_BLOCK_SIZE, 0, stream>>>
    (image_size, histogram_size, chunk_size, d_sgm_offset, d_sorted, d_hist);

  // synchronizes Kernel given a stream, instead of globally synchronizing on all threads
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaStreamSynchronize(stream) );
  //gpuErrchk( cudaDeviceSynchronize() );

 
  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);

  /* cudaFree(d_inds); */
  /* cudaFree(d_sgm_offset); */
  return elapsed;
}

// @ summary : Kernel launcher for asynchronous histogram starting
// @ remakrs : uses an argument structure defined in struct.cu.h
// @ params  : hist_args -> all the nececary information for a histogram kernel to run

template <class T>
void *launch_hist_kernel(void *hist_args)
{

  // converts argument to proper structure input
  hist_arg_struct<T> *args = (hist_arg_struct<T> *) hist_args;

  // waits for previous kernels to finish up work before copying.
  cudaEventSynchronize(args->event);

  // copy data from host to device for thread segment
  gpuErrchk( cudaMemcpyAsync(&args->d_image[args->offset], 
                             &args->h_image[args->global_offset],
                             args->image_size*sizeof(T), 
                             cudaMemcpyHostToDevice, 
                             args->stream));

  // computes histogram for thread segment, and commits to global device histogram
  unsigned long int elapsed = largeHistogramAsync<T>(args->image_size,
                                                     &args->d_image[args->offset],
                                                     args->d_inds,
                                                     args->d_sorted,
                                                     args->d_sgm_offset,
                                                     args->histogram_size,
                                                     args->d_hist,
                                                     args->boundary,
                                                     args->stream);
  // Records event, giving waiting thread permission to use device memory
  gpuErrchk( cudaEventRecord(args->stop_event, args->stream) );

  return NULL;
}


// @summary : Takes an array and returns and array of squared elements
// @remarks : Just give it a large input, and it will compute square of each element
// @params  : array_size     -> size of image
//          : h_in           -> input array
//          : h_out          -> for all h_out[i] = h_in[i]^2

template <class T>
void asyncHist(unsigned int  image_size,
               T*               h_image,
               unsigned int   hist_size,
               unsigned int*     h_hist,
               T             h_boundary){
  
  unsigned int stream_entries = 2;
  unsigned int stream_size = MAXIMUM_STREAM_SIZE;
  unsigned int nStreams = 0;
  unsigned int num_segments = ceil((float)hist_size / GPU_HIST_SIZE);


  while (nStreams < stream_entries) {
    stream_size = stream_size/2;
    nStreams = ceil((float)image_size / stream_size);
  }  

  unsigned int num_blocks = ceil((float)stream_size/CUDA_BLOCK_SIZE);
  unsigned int num_threads = nStreams;
  printf("num_threads %d\n", num_threads);
  pthread_t threads[num_threads];
  
  // Initailizes streams
  cudaStream_t stream[stream_entries];

  for (int i = 0; i < stream_entries; i++){
    gpuErrchk( cudaStreamCreate(&stream[i]) );
  }
  // Event management
  cudaEvent_t start[num_threads], stop[num_threads];

  for (int i = 0; i < num_threads; i++) {
    gpuErrchk( cudaEventCreate(&start[i]) );
    gpuErrchk( cudaEventCreate(&stop[i]) );
  }
  
  // Initializes device arrays
  T* d_image;
  unsigned int* d_inds, *d_sorted, *d_sgm_offset, *d_hist;
  gpuErrchk( cudaMalloc((void**) &d_image,     stream_size*stream_entries*sizeof(T)) );
  gpuErrchk( cudaMalloc((void**)  &d_inds,     stream_size*stream_entries*sizeof(int)) );
  gpuErrchk( cudaMalloc((void**)&d_sorted,     stream_size*stream_entries*sizeof(int)) );
  gpuErrchk( cudaMalloc((void**)&d_sgm_offset, num_segments*stream_entries*sizeof(int)) );
  gpuErrchk( cudaMalloc((void**)&d_hist, stream_size*stream_entries*sizeof(int)) );
  gpuErrchk( cudaMemset(d_hist, 0, sizeof(unsigned int)*hist_size) );

  // iniatilizes argument structure
  
  hist_arg_struct<T>* args[num_threads];
  for(int i = 0; i < num_threads; i++){
    args[i] = new hist_arg_struct<T>;
  }
  /* args[0]->image_size = 0; */
  /* printf("struct set : %d\n", args[0]->image_size); */

  struct timeval t_start, t_end;
  unsigned long int elapsed;
  gettimeofday(&t_start, NULL);

  // TODO : Figure if shared memory will be conflicting
  //      : Spawn threads and compute inputs for each thread.
  //      : Ensure that correct memory is passed to each Kernel
  //      : See if still correct when using asonchronous memcpy

  //unsigned int tid;
  for (int k = 0; k < num_threads; k++){
    printf("PROCESS : %d\n", k);
    if (k == nStreams-1){
      args[k]->image_size = image_size % stream_size;
    } else {
      args[k]->image_size = stream_size;
    }

    args[k]->h_image = h_image;
    args[k]->d_image = d_image;
    args[k]->boundary = h_boundary;
    args[k]->d_inds = &d_inds[(k%stream_entries)*stream_size];
    args[k]->d_sorted = &d_sorted[(k%stream_entries)*stream_size];
    args[k]->d_sgm_offset = &d_sgm_offset[(k%stream_entries)*num_segments];

    args[k]->histogram_size = hist_size;
    args[k]->d_hist = d_hist;
    args[k]->event = stop[k];
    if (k < (num_threads-2)){
      args[k]->stop_event = stop[k+2];
    } else {
      args[k]->stop_event = stop[k];
    }
    args[k]->stream = stream[k%stream_entries];
    args[k]->stream_size = stream_size;
    args[k]->offset = (k%stream_entries)*stream_size;
    args[k]->global_offset = k*stream_size;
    if (pthread_create(&threads[k], NULL, launch_hist_kernel<T>, (void*)args[k])) {
      fprintf(stderr, "Error creating threadn");
      return;
    }
    //gpuErrchk( cudaEventRecord(stop[k], stream[k]));
    /* if(pthread_join(threads[k], NULL)) { */
    /*   fprintf(stderr, "Error joining threadn"); */
    /*   return; */
    /* } */
  }
  printf("#############################\n");
  for (int i = 0; i< num_threads; i++){
    if(pthread_join(threads[i], NULL)) {
      fprintf(stderr, "Error joining threadn");
      return;
    }
  }

  for (int i = 0; i < stream_entries; i++){
    gpuErrchk( cudaEventRecord(stop[i], stream[i]));
  }



  // synchronizes all work done on GPU
  gpuErrchk( cudaDeviceSynchronize());

  gettimeofday(&t_end, NULL);
  elapsed = timeval_subtract(&t_end, &t_start);
  printf("asynchronous histogram (GPU) in %d µs\n", elapsed);

  // Copy back final histogram
  cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * hist_size, cudaMemcpyDeviceToHost);

  // Destroys streams once done
  for(int i = 0; i < stream_entries; i++){
    gpuErrchk( cudaStreamDestroy(stream[i]) );
  }

  // Destroy events
  for(int i = 0; i < num_threads; i++){
    gpuErrchk( cudaEventDestroy(start[i]) );
    gpuErrchk( cudaEventDestroy(stop[i]) );
  }

  // free device memory
  cudaFree(d_image);
  cudaFree(d_inds);
  cudaFree(d_sorted);
  cudaFree(d_sgm_offset);
  cudaFree(d_hist);

}

#endif //HOST_HIST

