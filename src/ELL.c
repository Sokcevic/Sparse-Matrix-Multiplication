#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint64_t rows;
  uint64_t cols;
  uint64_t nonZero;   // max nonZero in any column
  float *values;      // has cols * nonZero Elements
  uint64_t *indicies; // has cols * nonZero Elements
} ELLMatrix;

/**
 *
 * Converts a 2D Matrix into a ELLPACK struct and returns a pointer to the
 * struct
 *
 */
ELLMatrix *toELL(float **matrix, uint64_t m, uint64_t n) {

  ELLMatrix *result; // initialize
  result = (ELLMatrix *)malloc(sizeof(ELLMatrix));
  if (!result) {
    return NULL;
  }
  result->rows = m;
  result->cols = n;

  uint64_t nonZero[n];
  memset(nonZero, 0, sizeof(nonZero));
  // calculate max number of non Zero values per column
  for (uint64_t i = 0; i < m; i++) // Use Cache properly
    for (uint64_t j = 0; j < n; j++)
      if (matrix[i][j] != 0)
        nonZero[j]++;

  uint64_t max = 0;
  for (uint64_t i = 0; i < n; i++)
    if (nonZero[i] > max)
      max = nonZero[i];

  result->nonZero = max;

  result->values = malloc(max * n * sizeof(float));
  if (!result->values) {
    return NULL;
  }
  result->indicies = malloc(max * n * sizeof(uint64_t));
  if (!result->indicies) {
    return NULL;
  }

  uint64_t elements_added = 0;
  for (uint64_t i = 0; i < n; i++) { // iterate columns
    uint64_t added = 0;
    uint64_t dc = -1;
    for (uint64_t j = 0; j < m; j++) { // iterate rows
      if (matrix[j][i] != 0) {
        (result->values)[elements_added] = matrix[j][i];
        (result->indicies)[elements_added] = j;
        elements_added++;
        added++;
        if (elements_added % (max) == 0)
          break;
      } else {
        dc = j;
      }
    }
    for (; added < max; added++) {
      (result->values)[elements_added] = 0;
      (result->indicies)[elements_added] = dc;
      elements_added++;
    }
  }

  return result;
}

float **fromELL(ELLMatrix *m) {
  float **result;
  result = (float **)calloc(m->rows, sizeof(float *));
  if (!result)
    return NULL;

  for (int i = 0; i < m->rows; i++) {
    result[i] = calloc(m->cols, sizeof(float));
    if (!result)
      return NULL;
  }

  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    result[m->indicies[i]][i / m->nonZero] = m->values[i];
  }

  return result;
}

ELLMatrix *transpose(ELLMatrix *m) {
  float **result;
  result = (float **)calloc(m->cols, sizeof(float *));
  if (!result)
    return NULL;

  for (uint64_t i = 0; i < m->cols; i++) {
    result[i] = calloc(m->rows, sizeof(float));
    if (!result)
      return NULL;
  }

  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    result[i / m->nonZero][m->indicies[i]] = m->values[i];
  }

  ELLMatrix *ell = toELL(result, m->cols, m->rows);
  for (uint64_t i = 0; i < m->cols; i++) {
    free(result[i]);
  }
  free(result);
  return ell;
}
