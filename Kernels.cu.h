#ifndef KERNELS_HIST
#define KERNELS_HIST


template <class F, class T>
__global__ void mapIndKernel(unsigned int  tot_size,
                       unsigned int hist_size,
                       T             boundary,
                       T*                d_in,
                       int*             d_out){
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < tot_size) {
    d_out[gid] = (int)(d_in[gid]/boundary * (float)hist_size);
  }
}

#endif KERNELS_HIST //
