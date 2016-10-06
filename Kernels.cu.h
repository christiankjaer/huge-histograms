#ifndef KERNELS_HIST
#define KERNELS_HIST
#include <cuda_runtime.h>

template <class T>
__global__ void mapKer(unsigned int  tot_size,
											 unsigned int hist_size,
											 T             boundary,
											 T*                d_in,
											 int*             d_out){
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < tot_size) {
		// ... implement f for index corresponding to indices.
		d_out[gid] = floor(d_in[gid]/boundary) * hist_size;
	}
}

#endif KERNELS_HIST
