#ifndef KERNELS_HIST
#define KERNELS_HIST

#define GPU_HIST_SIZE 8192

template <class T>
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

__global__ void naiveHistKernel(unsigned int  tot_size,
                                int*              inds,
                                int*              hist) {

  __shared__ int Hsh[GPU_HIST_SIZE];


  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int bdx = blockDim.x;
  const unsigned int hist_elems = GPU_HIST_SIZE / bdx;
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

#endif //KERNELS_HIST
