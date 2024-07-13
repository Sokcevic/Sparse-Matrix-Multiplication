#ifndef ELLMATRIX_H
#define ELLMATRIX_H

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

/**
 * Converts a 2D Matrix into an ELLPACK struct and returns a pointer to the struct.
 *
 * @param matrix The input 2D matrix.
 * @param m The number of rows in the input matrix.
 * @param n The number of columns in the input matrix.
 * @return A pointer to the resulting ELLMatrix struct.
 */
ELLMatrix *toELL(float **matrix, uint64_t m, uint64_t n);

/**
 * Converts an ELLPACK struct back into a 2D matrix and returns a pointer to the matrix.
 *
 * @param m The input ELLMatrix struct.
 * @return A pointer to the resulting 2D matrix.
 */
float **fromELL(ELLMatrix *m);

/**
 * Transposes an ELLPACK struct and returns a pointer to the resulting transposed struct.
 *
 * @param m The input ELLMatrix struct.
 * @return A pointer to the resulting transposed ELLMatrix struct.
 */
ELLMatrix *transpose(ELLMatrix *m);

/**
 * Frees the memory allocated for an ELLMatrix struct.
 *
 * @param ell The ELLMatrix struct to free.
 */
void freeELL(ELLMatrix* ell);

#endif // ELLMATRIX_H

