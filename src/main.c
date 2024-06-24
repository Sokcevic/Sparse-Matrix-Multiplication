#include "ELL.c"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void matr_mult_ellpack(const void *a, const void *b, void *result) {
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
  float **mr = (float **)result;

  for (int i = 0; i < matrix_a->rows; i++) {
    for (int j = 0; j < matrix_b->cols; j++) {
      mr[i][j] += ma[i][j] * mb[j][i];
    }
  }

  result = mr;
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

  return EXIT_SUCCESS;
}
