#ifndef ARRAY_LIB

void fill_array(int* array, int array_length){
  for (int i = 0; i < array_length; i++){
    array[i] = (int)rand() % 30000;
  }
}

__inline__ void make_keys(int* A, int* K, int N, int MASK){
  for (int i = 0; i < N; i++){
    K[i] = A[i] & MASK;
  }
}

void print_array(int* array, int array_length){
  printf("[");
  int j = 0;
  for (int i = 0; i < array_length; i++){
    printf("%d", array[i]);
    if (i != array_length-1){
      printf(",");
      j = j;
      printf("\n");
    }
  }
  printf("].\n");
}

#endif //ARRAY_LIB
