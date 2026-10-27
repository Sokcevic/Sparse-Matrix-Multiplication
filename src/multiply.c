#include "multiply.h"
#include "ELL.h"
#include "matrix.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <xmmintrin.h>

/**
 * Dot product of column aCol of at (= row aCol of A) and column bCol of b.
 * Both columns are sorted by row index and end at the first padding value, so
 * matching indices are found by merging them.
 */
static float sparseDot(const ELLMatrix *at, uint64_t aCol, const ELLMatrix *b,
                       uint64_t bCol) {
  uint64_t a_index = aCol * at->nonZero;
  uint64_t b_index = bCol * b->nonZero;
  uint64_t a_end = a_index + at->nonZero;
  uint64_t b_end = b_index + b->nonZero;
  float sum = 0;
  while (a_index < a_end && b_index < b_end && at->values[a_index] != 0 &&
         b->values[b_index] != 0) {
    uint64_t a_row = at->indicies[a_index];
    uint64_t b_row = b->indicies[b_index];
    if (a_row == b_row) {
      sum += at->values[a_index] * b->values[b_index];
      a_index++;
      b_index++;
    } else if (a_row < b_row) {
      a_index++;
    } else {
      b_index++;
    }
  }
  return sum;
}

static void multiplyELL(const ELLMatrix *matrix_a, const ELLMatrix *matrix_b,
                        ELLMatrix *result, bool parallel) {
  ELLMatrix *at = transpose(matrix_a); // transpose to read through rows of a
  if (at == NULL) {
    return;
  }
  (void)parallel; // only used by the OpenMP pragmas

  // calculate nonZero of result: max number of non-zero values in a column
  uint64_t maxNonZero = 0;
#pragma omp parallel for reduction(max : maxNonZero) if (parallel)
  for (uint64_t i = 0; i < matrix_b->cols; i++) {
    uint64_t nonZero = 0;
    for (uint64_t j = 0; j < at->cols; j++) {
      if (sparseDot(at, j, matrix_b, i) != 0) {
        nonZero++;
      }
    }
    if (nonZero > maxNonZero) {
      maxNonZero = nonZero;
    }
  }

  ELLMatrix *res = createELL(at->cols, matrix_b->cols, maxNonZero);
  if (res == NULL) {
    freeELL(at);
    return;
  }

  // calculate the result column by column, padding stays 0 from calloc
#pragma omp parallel for if (parallel)
  for (uint64_t i = 0; i < matrix_b->cols; i++) {
    uint64_t result_index = i * res->nonZero;
    for (uint64_t j = 0; j < at->cols; j++) {
      float sum = sparseDot(at, j, matrix_b, i);
      if (sum != 0) {
        res->values[result_index] = sum;
        res->indicies[result_index] = j;
        result_index++;
      }
    }
  }

  *result = *res;
  free(res);
  freeELL(at);
}

// Ellpack parallel
void matr_mult_ellpack(const void *a, const void *b, void *result) {
  multiplyELL(a, b, result, true);
}

// Ellpack sequential
void matr_mult_ellpack_V1(const void *a, const void *b, void *result) {
  multiplyELL(a, b, result, false);
}

// normale MatMul SIMD
void matr_mult_ellpack_V2(const void *a, const void *b, void *result) {
  const ELLMatrix *matrix_a = a;
  ELLMatrix *matrix_b = transpose(b); // transpose to read through columns of b
  if (matrix_b == NULL) {
    return;
  }

  // cast to normal matrices
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  Matrix *mr = createMatrix(matrix_a->rows, matrix_b->rows);

  if (ma && mb && mr) {
    uint64_t n = matrix_a->cols;
    uint64_t simdEnd = n - n % 4;
    // iterate through entries of result and calculate them
    for (uint64_t i = 0; i < ma->rows; i++) {
      for (uint64_t j = 0; j < mb->rows; j++) {
        __m128 sum = _mm_setzero_ps();
        for (uint64_t k = 0; k < simdEnd; k += 4) {
          __m128 a_values = _mm_loadu_ps(&ma->data[i][k]);
          __m128 b_values = _mm_loadu_ps(&mb->data[j][k]);
          sum = _mm_add_ps(sum, _mm_mul_ps(a_values, b_values));
        }
        float temp[4];
        _mm_storeu_ps(temp, sum);
        float value = temp[0] + temp[1] + temp[2] + temp[3];
        // remaining values if n is not divisible by 4
        for (uint64_t k = simdEnd; k < n; k++) {
          value += ma->data[i][k] * mb->data[j][k];
        }
        mr->data[i][j] = value;
      }
    }

    ELLMatrix *mx = toELL(mr);
    if (mx) {
      *((ELLMatrix *)result) = *mx;
      free(mx);
    }
  }

  freeMatrix(ma);
  freeMatrix(mb);
  freeMatrix(mr);
  freeELL(matrix_b);
}
