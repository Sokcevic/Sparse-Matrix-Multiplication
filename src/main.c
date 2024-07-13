#include "ELL.h"
#include "matrix.h"
#include <stdint.h>
#include <stdlib.h>
#include <tmmintrin.h>

// Matmul parallel
void matr_mult_ellpack_v5(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = (ELLMatrix *)a;
  ELLMatrix *matrix_b = (ELLMatrix *)b;

  // Check if multiplication is possible
  if (matrix_a->cols != matrix_b->rows) {
    return; // Not allowed
  }

  // initialize result
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  Matrix *mr = createMatrix(ma->rows, mb->cols);

#pragma omp parallel for collapse(3)
  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->cols; j++) {
      for (uint64_t k = 0; k < matrix_a->cols; k++) {
#pragma omp atomic
        mr->data[i][j] += ma->data[i][k] * mb->data[k][j];
      }
    }
  }

  ELLMatrix *mx = toELL(mr);
  *((ELLMatrix *)result) = *mx;

  free(mx);
  freeMatrix(ma);
  freeMatrix(mb);
  freeMatrix(mr);
}

// Parallel SIMD
void matr_mult_ellpack_v4(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = (ELLMatrix *)a;
  ELLMatrix *matrix_b = transpose((ELLMatrix *)b);

  // Check if multiplication is possible
  if (matrix_a->cols != matrix_b->rows) {
    return; // Not allowed
  }

  // initialize result
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  Matrix *mr = createMatrix(ma->rows, mb->cols);

#pragma omp parallel for collapse(3)
  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->cols; j++) {
      if (matrix_a->cols > 3) {
        __m128 sum = _mm_setzero_ps(); // Initialize sum to zero
        for (uint64_t k = 0; k < matrix_a->cols; k += 4) {
          __m128 a_values = _mm_loadu_ps(&ma->data[i][k]);
          __m128 b_values = _mm_loadu_ps(&mb->data[j][k]);
          __m128 product = _mm_mul_ps(a_values, b_values);
          sum = _mm_add_ps(sum, product);
        }
        float temp[4];
        _mm_storeu_ps(temp, sum);
#pragma omp atomic
        mr->data[i][j] = temp[0] + temp[1] + temp[2] + temp[3];
      }
      for (int k = matrix_a->cols - (matrix_a->cols % 4); k < matrix_a->cols;
           k++) {
#pragma omp atomic
        mr->data[i][j] += ma->data[i][k];
      }
    }
  }
  ELLMatrix *mx = toELL(mr);
  *((ELLMatrix *)result) = *mx;

  free(mx);
  freeMatrix(ma);
  freeMatrix(mb);
  freeELL(matrix_b);
  freeMatrix(mr);
}

// SIMD Matmul
void matr_mult_ellpack_v3(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = (ELLMatrix *)a;
  ELLMatrix *matrix_b = transpose((ELLMatrix *)b);

  // Check if multiplication is possible
  if (matrix_a->cols != matrix_b->rows) {
    return; // Not allowed
  }

  // initialize result
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  Matrix *mr = createMatrix(ma->rows, mb->cols);

  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->cols; j++) {
      if (matrix_a->cols > 3) {
        __m128 sum = _mm_setzero_ps(); // Initialize sum to zero
        for (uint64_t k = 0; k < matrix_a->cols; k += 4) {
          __m128 a_values = _mm_loadu_ps(&ma->data[i][k]);
          __m128 b_values = _mm_loadu_ps(&mb->data[j][k]);
          __m128 product = _mm_mul_ps(a_values, b_values);
          sum = _mm_add_ps(sum, product);
        }
        float temp[4];
        _mm_storeu_ps(temp, sum);

        mr->data[i][j] = temp[0] + temp[1] + temp[2] + temp[3];
      }
      for (int k = matrix_a->cols - (matrix_a->cols % 4); k < matrix_a->cols;
           k++) {
        mr->data[i][j] += ma->data[i][k];
      }
    }
  }
  ELLMatrix *mx = toELL(mr);
  *((ELLMatrix *)result) = *mx;

  free(mx);
  freeMatrix(ma);
  freeMatrix(mb);
  freeELL(matrix_b);
  freeMatrix(mr);
}

// Matmul
void matr_mult_ellpack_v2(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = (ELLMatrix *)a;
  ELLMatrix *matrix_b = (ELLMatrix *)b;

  // Check if multiplication is possible
  if (matrix_a->cols != matrix_b->rows) {
    return; // Not allowed
  }

  // initialize result
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  Matrix *mr = createMatrix(ma->rows, mb->cols);

  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->cols; j++) {
      for (uint64_t k = 0; k < matrix_a->cols; k++) {
        mr->data[i][j] += ma->data[i][k] * mb->data[k][j];
      }
    }
  }

  ELLMatrix *mx = toELL(mr);
  *((ELLMatrix *)result) = *mx;

  free(mx);
  freeMatrix(ma);
  freeMatrix(mb);
  freeMatrix(mr);
}

