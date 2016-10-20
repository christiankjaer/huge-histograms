#ifndef KERNELS_HIST
#define KERNELS_HIST
#include "setup.cu.h"

// @summary : Computes the histogram indexes based on a normalization of the data
// @remarks : Assumes all values in the input array to be non-negative
// @params  : input_arr_d -> the input values
//          : hist_inds_d -> an array to write back the histogram indexes
//          : size_arr    -> the size of both arrays
//          : the largest input element
template <class T>
__global__ void histVals2IndexKernel(float* input_arr_d,
                                     int*   hist_inds_d,
                                     int    size_arr,
                                     float  max_input){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size_arr){
    hist_inds_d[gid] = (int)((input_arr_d[gid]/max_input)*(float)HISTOGRAM_SIZE);
  }
}

// @remarks : needs proper documentation
__global__ void naiveHistKernel(unsigned int  tot_size,
                                int*              inds,
                                int*              hist) {

  __shared__ int Hsh[CHUNK_SIZE];

  // Thread index
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // Block dimension
  const unsigned int bdx = blockDim.x;
  //
  const unsigned int hist_elems = CHUNK_SIZE / bdx;
  const unsigned int tot_elems = tot_size / bdx;

  if (gid < tot_size) {
    for (int i = 0; i < hist_elems; i++) {
      Hsh[i*bdx + gid] = 0;
    }
    __syncthreads();
    for (int i = 0; i < tot_elems; i++) {
      atomicAdd(&Hsh[inds[bdx*i + gid]], 1);
    }
    __syncthreads();
    for (int i = 0; i < hist_elems; i++) {
      hist[i*bdx + gid] = Hsh[i*bdx + gid];
    }
  }

}



// Following 4 kernels, are helpers for scan inclusive.

// @summary : computes one warp of each lane of scan_inc
template<class OP, class T>
  __device__ inline
  T scanIncWarp( volatile T* ptr, const unsigned int idx ) {
  const unsigned int lane = idx & 31;

  //  SIMD execution, hence no syncronizatoin needed!
  if (lane >= 1)  ptr[idx] = OP::apply(ptr[idx-1],  ptr[idx]);
  if (lane >= 2)  ptr[idx] = OP::apply(ptr[idx-2],  ptr[idx]);
  if (lane >= 4)  ptr[idx] = OP::apply(ptr[idx-4],  ptr[idx]);
  if (lane >= 8)  ptr[idx] = OP::apply(ptr[idx-8],  ptr[idx]);
  if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);

  return const_cast<T&>(ptr[idx]);
}

// @summary : helper kernel for scan inclusive
template<class OP, class T>
  __device__ inline
  T scanIncBlock(volatile T* ptr, const unsigned int idx) {
  const unsigned int lane   = idx &  31;
  const unsigned int warpid = idx >> 5;

  T val = scanIncWarp<OP,T>(ptr,idx);
  __syncthreads();

  // This works because (warp size)^2 = (max cuda block size)
  if (lane == 31) { ptr[warpid] = const_cast<T&>(ptr[idx]); }
  __syncthreads();

  if (warpid == 0) scanIncWarp<OP,T>(ptr, idx);
  __syncthreads();

  if (warpid > 0) {
    val = OP::apply(ptr[warpid-1], val);
  }

  return val;
}

// @summary : computes scan inclusive
template<class OP, class T>
  __global__ void
  scanIncKernel(T* d_in, T* d_out, unsigned int d_size) {
  extern __shared__ char sh_mem1[];
  volatile T* sh_memT = (volatile T*)sh_mem1;
  const unsigned int tid = threadIdx.x;
  const unsigned int gid = blockIdx.x*blockDim.x + tid;
  T el    = (gid < d_size) ? d_in[gid] : OP::identity();
  sh_memT[tid] = el;
  __syncthreads();
  T res   = scanIncBlock < OP, T >(sh_memT, tid);
  if (gid < d_size) d_out [gid] = res;
}

// @summary : helper kernel for scan inclusive
template<class T>
__global__ void
copyEndOfBlockKernel(T* d_in, T* d_out, unsigned int d_out_size) {
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

  if(gid < d_out_size)
    d_out[gid] = d_in[ blockDim.x*(gid+1) - 1];
}

// @summary : helper kernel for scan inclusive
template<class OP, class T>
  __global__ void
  distributeEndBlock(T* d_in, T* d_out, unsigned int d_size) {
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

  if(gid < d_size && blockIdx.x > 0)
    d_out[gid] = OP::apply(d_in[blockIdx.x-1], d_out[gid]);
}

#endif //KERNELS_HIST
