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

__host__ ELLMatrix *toELL(Matrix *matrix);
__host__ ELLMatrix *toDevice(ELLMatrix *matrix);
__host__ void freeFromDevice(ELLMatrix *d_matrix);
__host__ void copyFromDevice(ELLMatrix *src, ELLMatrix *target);
__host__ Matrix *fromELL(ELLMatrix *m);
__host__ ELLMatrix *transpose(ELLMatrix *m);
__host__ void freeELL(ELLMatrix *ell);
__host__ ELLMatrix *createELL(uint64_t rows, uint64_t cols, uint64_t nonZero);
__host__ void ellEquals(ELLMatrix *a, ELLMatrix *b, float *result);
__host__ void printEll(ELLMatrix *m);
__host__ ELLMatrix *generateRandomELL(uint64_t nonZero, uint64_t rows,
                                      uint64_t cols, float dcRate);

#endif // ELLMATRIX_H
