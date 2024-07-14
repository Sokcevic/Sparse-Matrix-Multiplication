#include "ELL.h"
#include "main.h"
#include "matrix.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

ELLMatrix *generateRandomELL(uint64_t nonZero, uint64_t rows, uint64_t cols,
                             float dcRate) {
  ELLMatrix *result = createELL(rows, cols, nonZero);
  if (!result) {
    return NULL;
  }
  for (uint64_t i = 0; i < cols; i++) {
    uint64_t dc = 0;
    if ((float)rand() / (float)RAND_MAX < dcRate) {
      dc = (rand() % nonZero);
    }
    for (uint64_t j = 0; j < nonZero - dc; j++) {
      if (j >= cols * nonZero) {
        printf(" ");
      }
      result->values[i * nonZero + j] =
          (1000.0F * ((float)rand() / (float)RAND_MAX)) +
          ((float)rand() / (float)RAND_MAX);
      result->indicies[i * nonZero + j] = j;
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
  int iterations = 100;
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

void testELLMul(uint64_t rows_a, uint64_t cols_a, uint64_t rows_b,
                uint64_t cols_b, float sparsity,
                void (*f)(const void *, const void *, void *)) {

  int iterations = 10;
  double times = 0;
  for (int i = 0; i < iterations; i++) {

    ELLMatrix *al = generateRandomELL(5, rows_a, cols_a, sparsity);
    ELLMatrix *bl = generateRandomELL(5, rows_b, cols_b, sparsity);
    ELLMatrix *cl = createELL(rows_a, cols_b, 0);

    printEll(al);
    printEll(bl);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    f(al, bl, cl);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double took =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    times += took;

    printf("took: %f\n", took);

    freeELL(al);
    freeELL(bl);
  }
  printf("avg time: %f", ((double)times) / ((double)iterations));
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
  test_matmul(300, 500, 500, 300, 0.3, &matr_mult_ellpack_v6);
}

int main(void) {

   //test_matmul_large();
  testELLMul(30000, 50000, 50000, 30000, 0.6, &matr_mult_ellpack_v7);
  // testELL();
  return EXIT_SUCCESS;
}
