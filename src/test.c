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

void test_matmul(uint64_t rows_a, uint64_t cols_a, uint64_t rows_b,
                 uint64_t cols_b, float sparsity,
                 void (*f)(const void *, const void *, void *)) {
  Matrix *a;
  Matrix *b;
  Matrix *c;
  int incorrect = 0;
  float maxdiff = 0.0;
  int iterations = 10;
  for (int i = 0; i < iterations; i++) {
    a = generateRandomMatrix(sparsity, rows_a, cols_a);
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

    float diff = matrixEqual(c, c2);
    freeMatrix(c);
    freeMatrix(c2);

    if (diff != 0) {
      incorrect++;
      maxdiff = maxdiff > diff ? maxdiff : diff;
    }
  }
  printf("%d/%d correct.\n", iterations - incorrect, iterations);
  printf("Max absolute difference: %f", maxdiff);
}

void testELL() {
  Matrix *a;
  Matrix *b;
  Matrix *c;
  int incorrect = 0;
  float maxdiff = 0.0;
  int iterations = 20;
  for (int i = 0; i < iterations; i++) {
    a = generateRandomMatrix(0.6, 300, 350);

    ELLMatrix *al = toELL(a);
    ELLMatrix *al_t = transpose(al);
    ELLMatrix *al_tt = transpose(al_t);

    Matrix *am = fromELL(al_tt);

    freeELL(al_tt);
    freeELL(al_t);

    float f = matrixEqual(am, a);
    printf("%f\n", f);

    freeMatrix(am);
    freeMatrix(a);
    freeELL(al);
  }
}

void test_matmul_large() {
  test_matmul(3000, 5000, 5000, 9000, 0.9, &matr_mult_ellpack_v4);
}

int main(void) {

  // test_matmul_large();
  testELL();
  return EXIT_SUCCESS;
}
