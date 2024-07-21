#include "matrix.cuh"
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__host__ void freeFromDevice(Matrix *m) {
  cudaFree(m->data);
  cudaFree(m);
}

__host__ Matrix *toDevice(Matrix *m) {
  Matrix *d_m;
  cudaError_t err = cudaMalloc((void **)&d_m, sizeof(Matrix));
  if (err != cudaSuccess) {
    fprintf(stderr, "Error allocating memory for Matrix: %s\n",
            cudaGetErrorString(err));
    return NULL;
  }

  float *d_data;
  err = cudaMalloc((void **)&d_data, m->rows * m->cols * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Error allocating memory for Matrix data: %s\n",
            cudaGetErrorString(err));
    cudaFree(d_m);
    return NULL;
  }
  err = cudaMemcpy(d_data, m->data, m->rows * m->cols * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error copying Matrix data to device: %s\n",
            cudaGetErrorString(err));
    cudaFree(d_data);
    cudaFree(d_m);
    return NULL;
  }

  Matrix h_m = *m;
  h_m.data = d_data;

  err = cudaMemcpy(d_m, &h_m, sizeof(Matrix), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error copying Matrix structure to device: %s\n",
            cudaGetErrorString(err));
    cudaFree(d_data);
    cudaFree(d_m);
    return NULL;
  }

  return d_m;
}

__host__ void copyFromDevice(Matrix *src, Matrix *target) {

  Matrix *m = (Matrix*)calloc(1, sizeof(Matrix));

  cudaError_t err = cudaMemcpy(m, src, sizeof(Matrix), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }
  target->cols = m->cols;
  target->rows = m->rows;
  err = cudaMemcpy(target->data, m->data,
                   target->cols * target->rows * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }
}

__host__ void freeMatrix(Matrix *a) {
  free(a->data);
  free(a);
}

__host__ void printMatrix(Matrix *m) {
  printf("\n");
  for (uint64_t i = 0; i < m->rows; i++) {
    printf("%f\t", m->data[i]);
    if (i % m->cols == 0) {
      printf("\n");
    }
  }
}

__global__ void matmul(Matrix *a, Matrix *b, Matrix *result) {
  // Calculate the row index of the result matrix
  uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  // Calculate the column index of the result matrix
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < a->rows && col < b->cols) {
    float sum = 0.0;
    for (uint64_t k = 0; k < a->cols; k++) {
      sum += a->data[row * a->cols + k] * b->data[k * b->cols + col];
    }
    result->data[row * result->cols + col] = sum;
  }
}

__host__ Matrix *createMatrix(uint64_t rows, uint64_t cols) {
  Matrix *result = (Matrix *)malloc(sizeof(Matrix));
  if (result == NULL) {
    return NULL; // TODO: Error handling
  }
  result->rows = rows;
  result->cols = cols;
  result->data = (float *)calloc(rows * cols, sizeof(float));
  if (result->data == NULL) {
    free(result);
    return NULL; // TODO: Error handling
  }
  return result;
}

__host__ float matrixEqual(Matrix *a, Matrix *b) {
  if (a == NULL || b == NULL) {
    return -1.0F; // TODO: Error handling
  }
  if (a->rows != b->rows || a->cols != b->cols) {
    return -1.0F; // TODO: Error handling
  }
  if (a->data == NULL || b->data == NULL) {
    fprintf(stderr, "Error: One or both data pointers are NULL.\n");
    return -1.0F; // TODO: Error handling
  }
  float max = 0.0F;
  for (uint64_t i = 0; i < a->rows * a->cols; i++) {
    if (a->data[i] != b->data[i]) {
      float value = a->data[i] - b->data[i];
      value = value > 0 ? value : -value;
      max = max > value ? max : value;
    }
  }
  return max;
}
