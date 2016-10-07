#ifndef HIST_CPU
#define HIST_CPU

void segmentSize(int *inds, int *hist, int size, int hist_size){
  int k = 0;
  int cur_seg = 8191;
  for (int i = 0; i < size; i++){
    if (inds[i] < cur_seg) {
      hist[k]++;
    } else {
      cur_seg += 8192;
      k++;
      hist[k]++;
    }
  }
}

#endif //HIST_CPU
