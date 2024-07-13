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

void setup() { srand(69); }

void test_matmul(uint64_t rows_a, uint64_t cols_a, uint64_t rows_b, uint64_t cols_b, float sparsity, void (*f)(const void*, const void*, void*)) {
  Matrix *a;
  Matrix *b;
  Matrix *c;
  for (int i = 0; i < 100; i++) {
    c = generateRandomMatrix(sparsity, rows_a, cols_a);
    ELLMatrix *cl = toELL(c);
    bool v = matrixEqual(c, fromELL(cl));
    if(!v){
    printf("fail!");
    return;
    }
    /**
    a = generateRandomMatrix(sparsity, rows_a , cols_a);
    b = generateRandomMatrix(sparsity, rows_b, cols_b);
    c = createMatrix(rows_a, cols_b);

    ELLMatrix *al = toELL(a);
    ELLMatrix *bl = toELL(b);
    ELLMatrix *cl = toELL(c);

    f(al, bl, cl);

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
    **/
  }
  printf("success!");
}

void test_matmul_large() {
  test_matmul(700, 500, 500, 900, 0.6, &matr_mult_ellpack_v3);
}


int main(void) {

  test_matmul_large();
  return EXIT_SUCCESS;
}
