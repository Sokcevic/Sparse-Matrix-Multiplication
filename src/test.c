#include <stdint.h>
#include <stdlib.h>

typedef struct{
  uint64_t rows;
  uint64_t cols;
  float** data;
}Matrix;

void freeMatrixStruct(Matrix* a){
  for (uint64_t i = 0; i < a -> rows; i++) {
    free(a -> data[i]);
  }
  free(a -> data);
  free(a);
}


void matmul(Matrix* a, Matrix* b, Matrix* result){
  for (uint64_t i = 0; i < a -> rows; i++) {
    for (uint64_t j = 0; j < b -> cols; j++) {
      for (uint64_t k = 0; k < a -> cols; k++) {
        result -> data[i][j] += a -> data[i][k] * b -> data[k][j];
      }
    }
  }
}



int main(void)
{

  return EXIT_SUCCESS;
}
