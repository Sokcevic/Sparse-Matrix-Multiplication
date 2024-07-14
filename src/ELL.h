#ifndef ELLMATRIX_H
#define ELLMATRIX_H

#include "matrix.h"
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

ELLMatrix *toELL(Matrix *matrix);
Matrix *fromELL(ELLMatrix *m);
ELLMatrix *transpose(ELLMatrix *m);
void freeELL(ELLMatrix *ell);
ELLMatrix *createELL(uint64_t rows, uint64_t cols, uint64_t nonZero);
float ellEquals(ELLMatrix *a, ELLMatrix *b);
void printEll(ELLMatrix *m);

#endif // ELLMATRIX_H
