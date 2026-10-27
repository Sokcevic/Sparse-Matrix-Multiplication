#include "ELL.h"
#include "matrix.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

ELLMatrix *createELL(uint64_t rows, uint64_t cols, uint64_t nonZero) {
  // cols * nonZero * sizeof(uint64_t) must fit into size_t
  if (nonZero != 0 && cols > SIZE_MAX / sizeof(uint64_t) / nonZero) {
    fprintf(stderr, "Matrix with %" PRIu64 " columns and %" PRIu64
                    " non-zero values per column is too large!\n",
            cols, nonZero);
    return NULL;
  }
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
  result->indicies = calloc(cols * nonZero, sizeof(uint64_t));
  result->values = calloc(cols * nonZero, sizeof(float));
  if (!result->indicies || !result->values) {
    fprintf(stderr, "Memory allocation failed!\n");
    freeELL(result);
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
ELLMatrix *toELL(const Matrix *matrix) {
  uint64_t m = matrix->rows;
  uint64_t n = matrix->cols;

  uint64_t *nonZero = calloc(n, sizeof(uint64_t));
  if (n != 0 && !nonZero) {
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  // calculate max number of non Zero values per column
  for (uint64_t i = 0; i < m; i++) // Use Cache properly
    for (uint64_t j = 0; j < n; j++)
      if (matrix->data[i][j] != 0)
        nonZero[j]++;

  uint64_t max = 0;
  for (uint64_t i = 0; i < n; i++)
    if (nonZero[i] > max)
      max = nonZero[i];
  free(nonZero);

  ELLMatrix *result = createELL(m, n, max);
  if (!result) {
    return NULL;
  }

  // padding slots are already 0 thanks to calloc
  for (uint64_t i = 0; i < n; i++) { // iterate columns
    uint64_t index = i * max;
    for (uint64_t j = 0; j < m; j++) { // iterate rows
      if (matrix->data[j][i] != 0) {
        result->values[index] = matrix->data[j][i];
        result->indicies[index] = j;
        index++;
      }
    }
  }

  return result;
}

Matrix *fromELL(const ELLMatrix *m) {
  Matrix *result = createMatrix(m->rows, m->cols);
  if (!result) {
    return NULL;
  }

  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    if (m->values[i] != 0) {
      result->data[m->indicies[i]][i / m->nonZero] = m->values[i];
    }
  }
  return result;
}

ELLMatrix *transpose(const ELLMatrix *m) {
  uint64_t size = m->cols * m->nonZero;
  // occurrences[r] = number of non-zero values in row r
  uint64_t *occurrences = calloc(m->rows, sizeof(uint64_t));
  if (m->rows != 0 && !occurrences) {
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  // calculate nonZero of transposed matrix
  uint64_t max_row_non_zero = 0;
  for (uint64_t i = 0; i < size; i++) {
    if (m->values[i] != 0) {
      uint64_t count = ++occurrences[m->indicies[i]];
      if (count > max_row_non_zero) {
        max_row_non_zero = count;
      }
    }
  }
  ELLMatrix *result = createELL(m->cols, m->rows, max_row_non_zero);
  if (!result) {
    free(occurrences);
    return NULL;
  }
  // write values and indices to transposed matrix. The columns of m are
  // visited in ascending order, so the indices of every row end up sorted.
  for (uint64_t r = 0; r < m->rows; r++) {
    occurrences[r] = 0;
  }
  for (uint64_t i = 0; i < size; i++) {
    if (m->values[i] == 0) {
      continue; // padding, its index is meaningless
    }
    uint64_t row = m->indicies[i];
    uint64_t index = max_row_non_zero * row + occurrences[row]++;
    result->values[index] = m->values[i];
    result->indicies[index] = i / m->nonZero;
  }
  free(occurrences);
  return result;
}

/**
 * Returns the maximum absolute difference between the values of a and b,
 * -1 if the dimensions differ and -2 if the positions of the values differ.
 */
float ellEquals(const ELLMatrix *a, const ELLMatrix *b) {
  if (a->cols != b->cols || a->rows != b->rows || a->nonZero != b->nonZero) {
    return -1.0F;
  }
  float max = 0.0F;
  for (uint64_t i = 0; i < a->cols * a->nonZero; i++) {
    if ((a->values[i] == 0) != (b->values[i] == 0))
      return -2.0F;
    if (a->values[i] != 0 && a->indicies[i] != b->indicies[i])
      return -2.0F;
    float val = a->values[i] - b->values[i];
    val = val < 0 ? -val : val;
    max = val > max ? val : max;
  }
  return max;
}

void printEll(const ELLMatrix *m) {
  printf("ELL: \n");
  printf("cols: %" PRIu64 " \n", m->cols);
  printf("rows: %" PRIu64 " \n", m->rows);
  printf("nonZero: %" PRIu64 " \n", m->nonZero);
  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    printf("%f,", m->values[i]);
  }
  printf("\n");
  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    printf("%" PRIu64 ",", m->indicies[i]);
  }
  printf("\n");
}

void freeELL(ELLMatrix *ell) {
  if (!ell)
    return;
  free(ell->indicies);
  free(ell->values);
  free(ell);
}

// uniformly distributed random number in [0, n)
static uint64_t randomBelow(uint64_t n) {
  uint64_t r = 0;
  for (int i = 0; i < 4; i++) { // RAND_MAX is only guaranteed to be 2^15 - 1
    r = (r << 16) ^ (uint64_t)rand();
  }
  return r % n;
}

static int compareIndices(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a;
  uint64_t y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

/**
 * Generates a random matrix with up to nonZero values per column at random
 * rows. With probability dcRate a column gets fewer values (padding).
 */
ELLMatrix *generateRandomELL(uint64_t nonZero, uint64_t rows, uint64_t cols,
                             float dcRate) {
  if (nonZero > rows) {
    nonZero = rows;
  }
  ELLMatrix *result = createELL(rows, cols, nonZero);
  if (!result || nonZero == 0) {
    return result;
  }
  for (uint64_t i = 0; i < cols; i++) {
    uint64_t dc = 0;
    if ((float)rand() / (float)RAND_MAX < dcRate) {
      dc = randomBelow(nonZero);
    }
    uint64_t count = nonZero - dc;
    uint64_t *indices = &result->indicies[i * nonZero];
    // sorted random values from [0, rows - count] plus their position are
    // sorted, distinct and smaller than rows
    for (uint64_t j = 0; j < count; j++) {
      indices[j] = randomBelow(rows - count + 1);
    }
    qsort(indices, count, sizeof(uint64_t), compareIndices);
    for (uint64_t j = 0; j < count; j++) {
      indices[j] += j;
      result->values[i * nonZero + j] =
          (1000.0F * ((float)rand() / (float)RAND_MAX)) +
          ((float)rand() / (float)RAND_MAX) + 1.0F; // never 0
    }
  }
  return result;
}
