#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    uint64_t rows;
    uint64_t cols;
    float** data;
} Matrix;

void freeMatrix(Matrix* a);
void printMatrix(Matrix* m); 
void matmul(Matrix* a, Matrix* b, Matrix* result);
Matrix *createMatrix(uint64_t rows, uint64_t cols);
bool matrixEqual(Matrix* a, Matrix* b);

#endif 
