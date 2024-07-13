#include "matrix.h"
#include <stdint.h>
#include <stdlib.h>

Matrix *generateRandomMatrix(float zeroRate, uint64_t rows, uint64_t cols) {
  Matrix *result = createMatrix(rows, cols);
  for (uint64_t i = 0; i < rows; i++) {
    for (uint64_t j = 0; j < cols; j++) {
      if ((float)rand() / (float)RAND_MAX < zeroRate) {
        result->data[i][j] = 0.0;
      } else {
        result->data[i][j] = (1000.0F * ((float)rand() / (float)RAND_MAX)) +
                             ((float)rand() / (float)RAND_MAX);
      }
    }
  }
  return result;
}

void setup() { srand(42); }

void test_matmul() {}

int main(void) {
  setup();
  Matrix *matrix = generateRandomMatrix(0.8F, 4, 5);
  printMatrix(matrix);
  freeMatrix(matrix);
  return EXIT_SUCCESS;
}
