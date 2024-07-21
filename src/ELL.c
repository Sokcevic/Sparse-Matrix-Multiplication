#include "ELL.h"
#include "matrix.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ELLMatrix *createELL(uint64_t rows, uint64_t cols, uint64_t nonZero) {
  ELLMatrix *result = malloc(sizeof(ELLMatrix));
  if (!result) {
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->rows = rows;
  result->cols = cols;
  result->nonZero = nonZero;
  if (nonZero == 0) {
    result->indicies = NULL;
    result->values = NULL;
    return result;
  }
  result->indicies = calloc(result->cols * result->nonZero, sizeof(uint64_t));
  if (!result->indicies) {
    free(result);
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->values = calloc(result->cols * result->nonZero, sizeof(float));
  if (!result->values) {
    free(result->indicies);
    free(result);
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  return result;
}

/**
 *
 * Converts a 2D Matrix into a ELLPACK struct and returns a pointer to the
 * struct
 *
 */
ELLMatrix *toELL(Matrix *matrix) {

  ELLMatrix *result; // initialize
  uint64_t m = matrix->rows;
  uint64_t n = matrix->cols;

  result = (ELLMatrix *)malloc(sizeof(ELLMatrix));
  if (!result) {
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->rows = m;
  result->cols = n;

  uint64_t nonZero[n];
  memset(nonZero, 0, sizeof(nonZero));
  // calculate max number of non Zero values per column
  for (uint64_t i = 0; i < m; i++) // Use Cache properly
    for (uint64_t j = 0; j < n; j++)
      if (matrix->data[i][j] != 0)
        nonZero[j]++;

  uint64_t max = 0;
  for (uint64_t i = 0; i < n; i++)
    if (nonZero[i] > max)
      max = nonZero[i];

  result->nonZero = max;
  __uint128_t size = (__uint128_t)max * (__uint128_t)n;

  if (size == 0) {
    result->indicies = NULL;
    result->values = NULL;
    return result;
  }

  result->values = (float *)calloc(size, sizeof(float));
  if (!result->values) {
    free(result);
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->indicies = (uint64_t *)calloc(size, sizeof(uint64_t));
  if (!result->indicies) {
    free(result->values);
    free(result);
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }

  uint64_t elements_added = 0;
  for (uint64_t i = 0; i < n; i++) { // iterate columns
    uint64_t added = 0;
    uint64_t dc = 0;
    for (uint64_t j = 0; j < m; j++) { // iterate rows
      if (matrix->data[j][i] != 0) {
        (result->values)[elements_added] = matrix->data[j][i];
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

Matrix *fromELL(ELLMatrix *m) {
  Matrix *result = createMatrix(m->rows, m->cols);

  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    if (m->values[i] != 0) {
      result->data[m->indicies[i]][i / m->nonZero] = m->values[i];
    }
  }
  return result;
}

ELLMatrix *transpose(ELLMatrix *m) {
  uint64_t *occurrences = NULL;
  if (m->nonZero != 0) {
    occurrences = (uint64_t *)calloc(m->rows, sizeof(uint64_t));
    if (!occurrences) {
      fprintf(stderr, "Memory allocation failed!\n");
      return NULL;
    }
  }
  // calculate nonZero of transposed matrix
  uint64_t max_row_non_zero = 0;
  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    if (m->values[i] != 0) {
      uint64_t idx = occurrences[m->indicies[i]];
      occurrences[m->indicies[i]] = ++idx;
      if (idx > max_row_non_zero) {
        max_row_non_zero = idx;
      }
    }
  }
  ELLMatrix *result = createELL(m->cols, m->rows, max_row_non_zero);
  if (!result) {
    if (occurrences)
      free(occurrences);
    return NULL;
  }
  // write values and indices to transposed matrix
  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    uint64_t index = max_row_non_zero * m->indicies[i];
    uint64_t batch = i / m->nonZero;
    for (; result->values[index] != 0; index++)
      ;
    // Next free index

    result->values[index] = m->values[i];
    result->indicies[index] = batch;
  }
  if (occurrences != NULL)
    free(occurrences);
  return result;
}

float ellEquals(ELLMatrix *a, ELLMatrix *b) {
  if (a->cols != b->cols || a->rows != b->rows || a->nonZero != b->nonZero) {
    return -1.0F;
  }
  float max = 0.0F;
  for (uint64_t i = 0; i < a->cols * a->nonZero; i++) {
    if (a->indicies[i] != b->indicies[i])
      return -2.0F;
    if (a->values[i] != b->values[i]) {
      float val = a->values[i] - b->values[i];
      val = val < 0 ? -val : val;
      max = val > max ? val : max;
    }
  }
  return max;
}

void printEll(ELLMatrix *m) {
  printf("ELL: \n");
  printf("cols: %lu \n", m->cols);
  printf("rows: %lu \n", m->rows);
  printf("nonZero: %lu \n", m->nonZero);
  for (int i = 0; i < m->cols * m->nonZero; i++) {
    printf("%f,", m->values[i]);
  }
  printf("\n");
  for (int i = 0; i < m->cols * m->nonZero; i++) {
    printf("%lu,", m->indicies[i]);
  }
}

void freeELL(ELLMatrix *ell) {
  if (ell->indicies)
    free(ell->indicies);
  if (ell->values)
    free(ell->values);
  free(ell);
}

ELLMatrix *generateRandomELL(uint64_t nonZero, uint64_t rows, uint64_t cols,
                             float dcRate) {
  ELLMatrix *result = createELL(rows, cols, nonZero);
  if (!result) {
    return NULL;
  }
  if (nonZero == 0) {
    return result;
  }
  for (uint64_t i = 0; i < cols; i++) {
    uint64_t dc = 0;
    if ((float)rand() / (float)RAND_MAX < dcRate) {
      dc = (rand() % nonZero);
    }
    for (uint64_t j = 0; j < nonZero - dc; j++) {
      if (j >= cols * nonZero) {
        break;
      }
      result->values[i * nonZero + j] =
          (1000.0F * ((float)rand() / (float)RAND_MAX)) +
          ((float)rand() / (float)RAND_MAX);
      result->indicies[i * nonZero + j] = j;
    }
  }
  return result;
}
