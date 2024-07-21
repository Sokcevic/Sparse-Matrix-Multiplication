#include "matrix.h"
#include <stdbool.h>
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
      float sum = 0.0;
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
    fprintf(stderr, "Memory allocation failed!");
    return NULL;
  }
  result->rows = rows;
  result->cols = cols;
  result->data = malloc(rows * sizeof(float *));
  if (result->data == NULL) {
    fprintf(stderr, "Memory allocation failed!");
    free(result);
    return NULL;
  }
  for (uint64_t i = 0; i < rows; i++) {
    result->data[i] = calloc(cols, sizeof(float));
    if (result->data[i] == NULL) {
      fprintf(stderr, "Memory allocation failed!");
      freeMatrix(result);
      return NULL;
    }
  }
  return result;
}

float matrixEqual(Matrix *a, Matrix *b) {
  if (a == NULL || b == NULL) {
    return -1.0F;//TODO: Error handling
  }
  if (a->rows != b->rows || a->cols != b->cols) {
    return -1.0F;//TODO: Error handling
  }
  float max = 0.0F;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      if (a->data[i][j] != b->data[i][j]) {
        float value = a->data[i][j] - b->data[i][j];
        value = value > 0 ? value : -value;
        max = max > value ? max : value;
      }
    }
  }
  return max;
}
