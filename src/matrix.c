#include "matrix.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void freeMatrix(Matrix *a) {
  if (!a)
    return;
  for (uint64_t i = 0; i < a->rows; i++) {
    free(a->data[i]);
  }
  free(a->data);
  free(a);
}

void printMatrix(const Matrix *m) {
  printf("\n");
  for (uint64_t i = 0; i < m->rows; i++) {
    for (uint64_t j = 0; j < m->cols; j++) {
      printf("%f\t", m->data[i][j]);
    }
    printf("\n");
  }
}

void matmul(const Matrix *a, const Matrix *b, Matrix *result) {
  for (uint64_t i = 0; i < a->rows; i++) {
    for (uint64_t j = 0; j < b->cols; j++) {
      float sum = 0.0F;
      for (uint64_t k = 0; k < a->cols; k++) {
        sum += a->data[i][k] * b->data[k][j];
      }
      result->data[i][j] = sum;
    }
  }
}

Matrix *createMatrix(uint64_t rows, uint64_t cols) {
  Matrix *result = malloc(sizeof(Matrix));
  if (result == NULL) {
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->rows = rows;
  result->cols = cols;
  result->data = calloc(rows, sizeof(float *));
  if (result->data == NULL) {
    fprintf(stderr, "Memory allocation failed!\n");
    free(result);
    return NULL;
  }
  for (uint64_t i = 0; i < rows; i++) {
    result->data[i] = calloc(cols, sizeof(float));
    if (result->data[i] == NULL) {
      fprintf(stderr, "Memory allocation failed!\n");
      freeMatrix(result);
      return NULL;
    }
  }
  return result;
}

float matrixEqual(const Matrix *a, const Matrix *b) {
  if (a == NULL || b == NULL) {
    return -1.0F;
  }
  if (a->rows != b->rows || a->cols != b->cols) {
    return -1.0F;
  }
  float max = 0.0F;
  for (uint64_t i = 0; i < a->rows; i++) {
    for (uint64_t j = 0; j < a->cols; j++) {
      if (a->data[i][j] != b->data[i][j]) {
        float value = a->data[i][j] - b->data[i][j];
        value = value > 0 ? value : -value;
        max = max > value ? max : value;
      }
    }
  }
  return max;
}
