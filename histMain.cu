#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Local headers
#include "Host.cu.h"
#include "histDataGen.cu.h"

int main () {
  const unsigned int row = 10;
  const unsigned int col = 10;
  float range = 100.0;
  const unsigned int mem_size = sizeof(float[row][col]);

  float *data = (float *) malloc(mem_size);
  
  genArray<float> (row*col, range, data);

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++){
      printf("%7.2lf", data[i*col+j]);
    }
    printf("\n");
  }


  return 0;
}