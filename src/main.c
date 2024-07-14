#include "ELL.h"
#include "matrix.h"
#include <stdint.h>
#include <stdlib.h>
#include <tmmintrin.h>

void matr_mult_ellpack_v6(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = transpose((ELLMatrix *)a);
  ELLMatrix *matrix_b = (ELLMatrix *)b;

  // Check if multiplication is possible
  if (matrix_a->rows != matrix_b->rows) {
    return; // Not allowed TODO: Error Handling
  }

  uint64_t maxNonZero = 0;
  for (uint64_t i = 0; i < matrix_b->cols; i++) {
    uint64_t nonZero = 0;
    for (uint64_t j = 0; j < matrix_a->cols; j++) {
      uint64_t a_batch_index = j * matrix_a->nonZero;
      uint64_t b_batch_index = i * matrix_b->nonZero;
      float sum = 0;
      while (a_batch_index < (j + 1) * matrix_a->nonZero &&
             b_batch_index < (i + 1) * matrix_b->nonZero) {
        if (matrix_a->values[a_batch_index] == 0 ||
            matrix_b->values[b_batch_index] == 0) {
          break;
        }
        uint64_t a_next_index = matrix_a->indicies[a_batch_index];
        uint64_t b_next_index = matrix_b->indicies[b_batch_index];
        if (a_next_index == b_next_index) {
          sum +=
              matrix_a->values[a_batch_index] * matrix_b->values[b_batch_index];
          a_batch_index++;
          b_batch_index++;
        } else if (a_next_index < b_next_index) {
          a_batch_index++;
        } else {
          b_batch_index++;
        }
      }
      if (sum != 0) {
        nonZero++;
      }
    }
    maxNonZero = maxNonZero > nonZero ? maxNonZero : nonZero;
  }

  ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, maxNonZero);
  if (!res) {
    freeELL(matrix_a);
    return;
  }

  uint64_t result_index = 0;
  for (uint64_t i = 0; i < matrix_b->cols; i++) {
    result_index = i * res->nonZero;
    for (uint64_t j = 0; j < matrix_a->cols; j++) {
      uint64_t a_batch_index = j * matrix_a->nonZero;
      uint64_t b_batch_index = i * matrix_b->nonZero;
      float sum = 0;
      while (a_batch_index < (j + 1) * matrix_a->nonZero &&
             b_batch_index < (i + 1) * matrix_b->nonZero) {
        if (matrix_a->values[a_batch_index] == 0 ||
            matrix_b->values[b_batch_index] == 0) {
          break;
        }
        uint64_t a_next_index = matrix_a->indicies[a_batch_index];
        uint64_t b_next_index = matrix_b->indicies[b_batch_index];
        if (a_next_index == b_next_index) {
          sum +=
              matrix_a->values[a_batch_index] * matrix_b->values[b_batch_index];
          a_batch_index++;
          b_batch_index++;
        } else if (a_next_index < b_next_index) {
          a_batch_index++;
        } else {
          b_batch_index++;
        }
      }
      if (sum != 0) {
        res->values[result_index] = sum;
        res->indicies[result_index] = j;
        result_index++;
      }
    }
  }

  *((ELLMatrix *)result) = *res;

  free(res);
  freeELL(matrix_a);
}

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
  if (matrix_a->cols != matrix_b->cols) {
    freeELL(matrix_b);
    return; // Not allowed
  }

  // initialize result
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  Matrix *mr = createMatrix(ma->rows, mb->rows);

#pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->rows; j++) {
      if (matrix_a->cols > 3) {
        __m128 sum = _mm_setzero_ps(); // Initialize sum to zero
        for (uint64_t k = 0; k < matrix_a->cols - (matrix_a->cols % 4);
             k += 4) {
          __m128 a_values = _mm_loadu_ps(&ma->data[i][k]);
          __m128 b_values = _mm_loadu_ps(&mb->data[j][k]);
          __m128 product = _mm_mul_ps(a_values, b_values);
          sum = _mm_add_ps(sum, product);
        }
        float temp[4];
        _mm_storeu_ps(temp, sum);
#pragma omp atomic
        mr->data[i][j] += temp[0] + temp[1] + temp[2] + temp[3];
      }
      for (int k = matrix_a->cols - (matrix_a->cols % 4); k < matrix_a->cols;
           k++) {
        mr->data[i][j] += ma->data[i][k] * mb->data[j][k];
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
  if (matrix_a->cols != matrix_b->cols) {
    freeELL(matrix_b);
    return; // Not allowed
  }

  // initialize result
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  Matrix *mr = createMatrix(ma->rows, mb->rows);

  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->rows; j++) {
      if (matrix_a->cols > 3) {
        __m128 sum = _mm_setzero_ps(); // Initialize sum to zero
        for (uint64_t k = 0; k < matrix_a->cols - (matrix_a->cols % 4);
             k += 4) {
          __m128 a_values = _mm_loadu_ps(&ma->data[i][k]);
          __m128 b_values = _mm_loadu_ps(&mb->data[j][k]);
          __m128 product = _mm_mul_ps(a_values, b_values);
          sum = _mm_add_ps(sum, product);
        }
        float temp[4];
        _mm_storeu_ps(temp, sum);

        mr->data[i][j] += temp[0] + temp[1] + temp[2] + temp[3];
      }
      for (int k = matrix_a->cols - (matrix_a->cols % 4); k < matrix_a->cols;
           k++) {
        mr->data[i][j] += ma->data[i][k] * mb->data[j][k];
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
