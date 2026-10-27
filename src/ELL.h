#ifndef ELLMATRIX_H
#define ELLMATRIX_H

#include "matrix.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ELLPACK (column-major) matrix.
 *
 * Column c occupies the slots [c * nonZero, (c + 1) * nonZero) of values and
 * indicies. Every function in this project keeps the following invariant:
 * the non-zero entries of a column come first, sorted by strictly increasing
 * row index, and all remaining slots are padding with value 0.
 */
typedef struct {
  uint64_t rows;
  uint64_t cols;
  uint64_t nonZero;   // max nonZero in any column
  float *values;      // has cols * nonZero Elements
  uint64_t *indicies; // has cols * nonZero Elements
} ELLMatrix;

ELLMatrix *toELL(const Matrix *matrix);
Matrix *fromELL(const ELLMatrix *m);
ELLMatrix *transpose(const ELLMatrix *m);
void freeELL(ELLMatrix *ell);
ELLMatrix *createELL(uint64_t rows, uint64_t cols, uint64_t nonZero);
float ellEquals(const ELLMatrix *a, const ELLMatrix *b);
void printEll(const ELLMatrix *m);
ELLMatrix *generateRandomELL(uint64_t nonZero, uint64_t rows, uint64_t cols,
                             float dcRate);

#ifdef __cplusplus
}
#endif

#endif // ELLMATRIX_H
