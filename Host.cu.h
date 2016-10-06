#ifndef HOST_HIST
#define HOST_HIST

#include "Kernels.cu.h"


template <class T>
void arraySgm (unsigned int      d_size,
               unsigned int d_hist_size,
               unsigned int  block_size,
               T               boundary,
               T*                  d_in,
               T*                 d_out) {
  int num_blocks = ceil(d_size / block_size);

  mapKer<T><<<num_blocks, block_size>>>(d_size, d_hist_size, boundary, d_in, d_out);
}

#endif HOST_HIST
