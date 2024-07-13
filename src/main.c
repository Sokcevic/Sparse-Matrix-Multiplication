#include "ELL.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tmmintrin.h>

void freeMatrix(float **m, uint64_t rows) {
  for (int i = 0.0; i < rows; i++) {
    free(m[i]);
  }
  free(m);
}

void matr_mult_ellpack(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = (ELLMatrix *)a;
  ELLMatrix *matrix_b = transpose((ELLMatrix *)b);

  // Check if multiplication is possible
  if (matrix_a->cols != matrix_b->rows) {
    return; // Not allowed
  }

  // initialize result
  float **ma = fromELL(matrix_a);
  float **mb = fromELL(matrix_b);
  float **mr = calloc(matrix_a->rows, sizeof(float *));
  if (mr == NULL) {
    return;
  }
  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    mr[i] = calloc(matrix_b->cols, sizeof(float));
    if (mr[i] == NULL) {
      freeMatrix(ma, matrix_a->rows);
      freeMatrix(mb, matrix_b->rows);
      freeMatrix(mr, matrix_a->rows);
      return;
    }
  }

  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->cols; j++) {
      if (matrix_a->cols > 3) {
        __m128 sum = _mm_setzero_ps(); // Initialize sum to zero
        for (uint64_t k = 0; k < matrix_a->cols; k += 4) {
          __m128 a_values = _mm_loadu_ps(&ma[i][k]);
          __m128 b_values = _mm_loadu_ps(&mb[j][k]);
          __m128 product = _mm_mul_ps(a_values, b_values);
          sum = _mm_add_ps(sum, product);
        }
        float temp[4];
        _mm_storeu_ps(temp, sum);

        mr[i][j] = temp[0] + temp[1] + temp[2] + temp[3];
      }
      for (int k = matrix_a->cols - (matrix_a->cols % 4); k < matrix_a->cols;
           k++) {
        mr[i][j] += ma[i][k];
      }
    }
  }
  *((ELLMatrix *)result) = *toELL(mr, matrix_a->rows, matrix_b->cols);

  freeMatrix(ma, matrix_a->rows);
  freeMatrix(mb, matrix_b->rows);
  freeMatrix(mr, matrix_a->rows);
}

void matr_mult_ellpack_v2(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = (ELLMatrix *)a;
  ELLMatrix *matrix_b = (ELLMatrix *)b;

  // Check if multiplication is possible
  if (matrix_a->cols != matrix_b->rows) {
    return; // Not allowed
  }

  // initialize result
  float **ma = fromELL(matrix_a);
  float **mb = fromELL(matrix_b);
  float **mr = calloc(matrix_a->rows, sizeof(float *));
  if (mr == NULL) {
    return;
  }
  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    mr[i] = calloc(matrix_b->cols, sizeof(float));
    if (mr[i] == NULL) {
      freeMatrix(ma, matrix_a->rows);
      freeMatrix(mb, matrix_b->rows);
      freeMatrix(mr, matrix_a->rows);
      return;
    }
  }

  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->cols; j++) {
      for (uint64_t k = 0; k < matrix_a->cols; k++) {
        mr[i][j] += ma[i][k] * mb[k][j];
      }
    }
  }

  *((ELLMatrix *)result) = *toELL(mr, matrix_a->rows, matrix_b->cols);

  freeMatrix(ma, matrix_a->rows);
  freeMatrix(mb, matrix_b->rows);
  freeMatrix(mr, matrix_a->rows);
}

int main(void) {
  // initialize
  uint64_t m = 4;
  uint64_t n = 4;
  float **matrix;
  float a[4][4] = {{5, 0, 0, 0}, {0, 6, 0, 0}, {0.5, 7, 0, 0}, {0, 0, 0, 3}};

  matrix = (float **)malloc(m * sizeof(float *));
  if (!matrix) {
    return -1;
  }
  for (int i = 0; i < 4; i++) {
    matrix[i] = (float *)malloc(n * sizeof(float));
    if (!matrix[i]) {
      return -1;
    }
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i][j] = a[i][j];
    }
  }

  // Initialization Done
  // Convert to ell
  ELLMatrix *ell = toELL(matrix, m, n);
  if (!ell) {
    return -1;
  }
  printf("ELLMATRIX: \n");
  printf("matrix dimensions: %lu x %lu \n", ell->rows, ell->cols);
  printf("NoNonzero: %lu\n", ell->nonZero);
  printf("values: ");
  ELLMatrix *result = calloc(1, sizeof(ELLMatrix));

  matr_mult_ellpack(ell, ell, result);
  printf("\n");
  float **undo = fromELL(result);
  if (!undo) {
    return -1;
  }

  printf("Undo: \n\n");

  for (uint64_t i = 0; i < ell->rows; i++) {
    for (uint64_t j = 0; j < ell->cols; j++) {
      printf("%f\t", undo[i][j]);
    }
    printf("\n");
  }

  /**
    for (int i = 0; i < (ell->cols) * (ell->nonZero); i++) {
      printf("%f, ", (ell->values)[i]);
    }
    printf("\n");

    printf("indices: ");
    for (int i = 0; i < (ell->cols) * (ell->nonZero); i++) {
      printf("%lu, ", (ell->indicies)[i]);
    }
    printf("\n");

    float **undo = fromELL(ell);
    if (!undo) {
      return -1;
    }

    printf("Undo: \n\n");

    for (uint64_t i = 0; i < ell->rows; i++) {
      for (uint64_t j = 0; j < ell->cols; j++) {
        printf("%f\t", undo[i][j]);
      }
      printf("\n");
    }

    printf("Transpose: \n\n");
    ELLMatrix *transposed = transpose(ell);
    if (!transposed) {
      return -1;
    }
    undo = fromELL(transposed);

    for (uint64_t i = 0; i < transposed->rows; i++) {
      for (uint64_t j = 0; j < transposed->cols; j++) {
        printf("%f\t", undo[i][j]);
      }
      printf("\n");
    }

    // free space
    for (int i = 0; i < m; i++) {
      free(*(matrix + i));
      free(undo[i]);
    }
    free(undo);
    free(matrix);

    free(ell->indicies);
    free(ell->values);
    free(ell);
    free(transposed->values);
    free(transposed->indicies);
    free(transposed);
    **/

  return EXIT_SUCCESS;
}
