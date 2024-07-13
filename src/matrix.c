#include "matrix.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void freeMatrix(Matrix *a) {
  for (uint64_t i = 0; i < a->rows; i++) {
    free(a->data[i]);
  }
  free(a->data);
  free(a);
}

void printMatrix(Matrix *m) {
  printf("\n");
  for (uint64_t i = 0; i < m->rows; i++) {
    for (uint64_t j = 0; j < m->cols; j++) {
      printf("%f\t", m->data[i][j]);
    }
    printf("\n");
  }
}

void matmul(Matrix *a, Matrix *b, Matrix *result) {
  for (uint64_t i = 0; i < a->rows; i++) {
    for (uint64_t j = 0; j < b->cols; j++) {
      for (uint64_t k = 0; k < a->cols; k++) {
        result->data[i][j] += a->data[i][k] * b->data[k][j];
      }
    }
  }
}

Matrix *createMatrix(uint64_t rows, uint64_t cols) {
  Matrix *result = malloc(sizeof(Matrix));
  if (result == NULL) {
    return NULL;
  }
  result->rows = rows;
  result->cols = cols;
  result->data = malloc(rows * sizeof(float *));
  if (result->data == NULL) {
    free(result);
    return NULL;
  }
  for (uint64_t i = 0; i < rows; i++) {
    result->data[i] = calloc(cols, sizeof(float));
    if (result->data[i] == NULL) {
      freeMatrix(result);
      return NULL;
    }
  }
  return result;
}
