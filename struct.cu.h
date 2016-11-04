#ifndef ARG_STRUCT
#define ARG_STRUCT
#include <pthread.h>

// structures to pass arguments to thread when running asynchronous histogram computation
template <class T> 
struct hist_arg_struct{
  unsigned int image_size;     // size of image
  T* h_image;                  // host actual image
  T* d_image;                  // device actual image
  T boundary;                  // boundary element
  unsigned int *d_inds;        // tmp index array
  unsigned int *d_sorted;      // sorted index array
  unsigned int *d_sgm_offset;  // segment off-set array
  unsigned int histogram_size; // histogram size
  unsigned int *d_hist;        // device histogram array
  cudaEvent_t event;           // thread event
  cudaEvent_t event_mem;       // thread event
  cudaEvent_t stop_event_mem;  // thread event
  cudaEvent_t stop_event;      // thread event
  cudaStream_t stream;         // thread stream
  unsigned int offset;         // stream offset
  unsigned int global_offset;  // stream offset
  pthread_mutex_t *mutex;       // lock
};// hist_arg_struct;

#endif // ARG_STRUCT
