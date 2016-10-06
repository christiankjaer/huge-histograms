#ifndef DATA_GEN
#define DATA_GEN

template <class T>
void genArray (int    size,
               T     range,
               T*  dataArr){
  for (int i = 0; i < size; i++){
    dataArr[i] = (T) rand()/ ((T) RAND_MAX() * range);
  }
}

#endif DATA_GEN
