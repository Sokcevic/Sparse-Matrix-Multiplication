#ifndef ELLMATRIX_H
#define ELLMATRIX_H

#include "matrix.cuh"
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

__host__ void toELL(Matrix *matrix, ELLMatrix* result);
__host__ ELLMatrix *ELLtoDevice(ELLMatrix *matrix);
__host__ void freeFromDevice(ELLMatrix *d_matrix);
__host__ void copyFromDevice(ELLMatrix *src, ELLMatrix* target);
__global__ void fromELL(ELLMatrix *m, Matrix* result);
__global__ void transpose(ELLMatrix *m);
__host__ void freeELL(ELLMatrix *ell);
__host__ void createELL(uint64_t rows, uint64_t cols, uint64_t nonZero, ELLMatrix* result);
__host__ void ellEquals(ELLMatrix *a, ELLMatrix *b, float* result);
__host__ void printEll(ELLMatrix *m);

#endif // ELLMATRIX_H
