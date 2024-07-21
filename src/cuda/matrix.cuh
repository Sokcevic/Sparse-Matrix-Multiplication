#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    uint64_t rows;
    uint64_t cols;
    float* data;
} Matrix;

__host__ void freeMatrix(Matrix* a);
__host__ void printMatrix(Matrix* m); 
__global__ void matmul(Matrix* a, Matrix* b, Matrix* result);
__host__ Matrix *createMatrix(uint64_t rows, uint64_t cols);
__host__ float matrixEqual(Matrix* a, Matrix* b);
__host__ Matrix* toDevice(Matrix* m);
__host__ void copyFromDevice(Matrix* src, Matrix* target);
__host__ void freeFromDevice(Matrix* m);

#endif 
