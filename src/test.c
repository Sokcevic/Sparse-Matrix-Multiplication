#include "ELL.h"
#include "main.h"
#include "matrix.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
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

void test_matmul() {
  Matrix *a;
  Matrix *b;
  Matrix *c;
  for (int i = 0; i < 10; i++) {
    a = generateRandomMatrix(0.6, 17, 21);
    b = generateRandomMatrix(0.6, 21, 13);
    c = createMatrix(17, 13);

    ELLMatrix *al = toELL(a);
    ELLMatrix *bl = toELL(b);
    ELLMatrix *cl = toELL(c);

    matr_mult_ellpack_v2(al, bl, cl);

    freeELL(al);
    freeELL(bl);

    Matrix *c2 = fromELL(cl);

    freeELL(cl);

    matmul(a, b, c);
    freeMatrix(a);
    freeMatrix(b);

    bool equal = matrixEqual(c, c2);
    freeMatrix(c);
    freeMatrix(c2);

    if (!equal) {
      printf("fail!");
      return;
    }
  }
  printf("success!");
}

int main(void) {

  test_matmul();
  return EXIT_SUCCESS;
}
