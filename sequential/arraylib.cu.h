#ifndef ARRAY_LIB
#define MAX_RANDOM_NUMBER_SIZE 30000

void fill_array(int* array, int array_length){
  for (int i = 0; i < array_length; i++){
    array[i] = (int)rand() % MAX_RANDOM_NUMBER_SIZE;
  }
}

void zero_array(int* array, int array_length){
  for (int i = 0; i < array_length; i++){
    array[i] = 0;
  }
}

void scan_exc(int* array, int array_length){
  for (int i = 1; i < array_length; i++){
    array[i] = array[i-1];
  }
  array[0] = 0;
  for (int i = 1; i < array_length; i++){
    array[i] = array[i-1] + array[i];
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
    printf("%6d", array[i]);
    if (i != array_length-1){
      printf(",");
      if (j == 10) {printf("\n "); j = 0;}
      else{j++;}
    }
  }
  printf("].\n");
}

#endif //ARRAY_LIB
