#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t rows;
    uint64_t cols;
    float** data;
} Matrix;

void freeMatrix(Matrix* a);
void printMatrix(const Matrix* m);
void matmul(const Matrix* a, const Matrix* b, Matrix* result);
Matrix *createMatrix(uint64_t rows, uint64_t cols);
float matrixEqual(const Matrix* a, const Matrix* b);

#ifdef __cplusplus
}
#endif

#endif 
