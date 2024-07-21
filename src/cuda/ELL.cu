#include "ELL.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

__host__ void copyFromDevice(ELLMatrix *src, ELLMatrix *target) {
  ELLMatrix *m = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  cudaError_t err =
      cudaMemcpy(m, src, sizeof(ELLMatrix), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }

  target->cols = m->cols;
  target->rows = m->rows;
  target->nonZero = m->nonZero;

  if (!(target->values)) {
    target->values =
        (float *)calloc(target->cols * target->nonZero, sizeof(float));
    if (!(target->values)) {
      fprintf(stderr, "Could not allocate memory.");
      return;
    }
  }

  err = cudaMemcpy(target->values, m->values,
                   target->cols * target->nonZero * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }

  if (!(target->indicies)) {
    target->indicies =
        (uint64_t *)calloc(target->cols * target->nonZero, sizeof(uint64_t));
    if (!(target->indicies)) {
      fprintf(stderr, "Could not allocate memory.");
      return;
    }
  }

  err = cudaMemcpy(target->indicies, m->indicies,
                   target->cols * target->nonZero * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }
}

__host__ ELLMatrix *toDevice(ELLMatrix *matrix) {
  ELLMatrix *d_matrix;
  cudaError_t cudaError = cudaMalloc((void **)&d_matrix, sizeof(ELLMatrix));
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't allocate memory for indices on the GPU: %s\n",
            cudaGetErrorString(cudaError));
    return NULL;
  }
  float *d_values;
  cudaError = cudaMalloc((void **)&d_values,
                         matrix->cols * matrix->nonZero * sizeof(float));

  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't allocate memory for indices on the GPU: %s\n",
            cudaGetErrorString(cudaError));
    cudaFree(d_matrix);
    return NULL;
  }

  uint64_t *d_indicies;
  cudaError = cudaMalloc((void **)&d_indicies,
                         matrix->cols * matrix->nonZero * sizeof(uint64_t));

  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't allocate memory for indices on the GPU: %s\n",
            cudaGetErrorString(cudaError));
    cudaFree(d_values);
    cudaFree(d_matrix);
    return NULL;
  }
  if (matrix->indicies == NULL || matrix->values == NULL) {
    ELLMatrix m = *matrix;
    m.indicies = d_indicies;
    m.values = d_values;
    cudaMemcpy(d_matrix, &m, sizeof(ELLMatrix), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "Couldn't copy ELLMatrix to the GPU: %s\n",
              cudaGetErrorString(cudaError));
      cudaFree(d_indicies);
      cudaFree(d_values);
      cudaFree(d_matrix);
      return NULL;
    }
    return d_matrix;
  }
  cudaMemcpy(d_values, matrix->values,
             matrix->cols * matrix->nonZero * sizeof(float),
             cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't copy ELLMatrix to the GPU: %s\n",
            cudaGetErrorString(cudaError));
    cudaFree(d_indicies);
    cudaFree(d_values);
    cudaFree(d_matrix);
    return NULL;
  }
  cudaMemcpy(d_indicies, matrix->indicies,
             matrix->cols * matrix->nonZero * sizeof(uint64_t),
             cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't copy ELLMatrix to the GPU: %s\n",
            cudaGetErrorString(cudaError));
    cudaFree(d_indicies);
    cudaFree(d_values);
    cudaFree(d_matrix);
    return NULL;
  }

  ELLMatrix m = *matrix;
  m.indicies = d_indicies;
  m.values = d_values;
  cudaMemcpy(d_matrix, &m, sizeof(ELLMatrix), cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't copy ELLMatrix to the GPU: %s\n",
            cudaGetErrorString(cudaError));
    cudaFree(d_indicies);
    cudaFree(d_values);
    cudaFree(d_matrix);
    return NULL;
  }
  return d_matrix;
}

__global__ void countNonZero(Matrix *matrix, uint64_t *nonZero) {
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t *colCount = (uint64_t *)malloc(matrix->cols * sizeof(uint64_t));
  if (!colCount) {
    return;
  }

  if (col < matrix->cols) {
    uint64_t count = 0;
    for (uint64_t row = 0; row < matrix->rows; row++) {
      if (matrix->data[row * matrix->cols + col] != 0) {
        count++;
      }
    }
    colCount[col] = count;
  }
  __syncthreads();
  atomicMax((unsigned long long *)nonZero, colCount[col]);
  return;
}

__host__ void freeFromDevice(ELLMatrix *d_matrix) {
  cudaFree(d_matrix->indicies);
  cudaFree(d_matrix->values);
  cudaFree(d_matrix);
}
__host__ void createELL(uint64_t rows, uint64_t cols, uint64_t nonZero,
                        ELLMatrix *result) {
  result->rows = rows;
  result->cols = cols;
  result->nonZero = nonZero;
  if (nonZero == 0) {
    result->indicies = NULL;
    result->values = NULL;
    return;
  }
  result->indicies =
      (uint64_t *)calloc(result->cols * result->nonZero, sizeof(uint64_t));
  if (!result->indicies) {
    free(result);
    return; // TODO: Error Handling
  }
  result->values =
      (float *)calloc(result->cols * result->nonZero, sizeof(float));
  if (!result->values) {
    free(result->indicies);
    free(result);
    return; // TODO: Error Handling
  }
  return;
}

__host__ ELLMatrix *transpose(ELLMatrix *m) {
  uint64_t *occurrences = (uint64_t *)calloc(m->rows, sizeof(uint64_t));
  if (!occurrences) {
    return NULL; // TODO: Error handling
  }
  uint64_t max_row_non_zero = 0;
  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    if (m->values[i] != 0) {
      uint64_t idx = occurrences[m->indicies[i]];
      occurrences[m->indicies[i]] = ++idx;
      if (idx > max_row_non_zero) {
        max_row_non_zero = idx;
      }
    }
  }
  ELLMatrix *result = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  if (!result) {
    fprintf(stderr, "Error allocation memory");
    return NULL;
  }
  createELL(m->cols, m->rows, max_row_non_zero, result);
  if (!result) {
    free(occurrences);
    return NULL; // TODO: Error Handling
  }
  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    uint64_t index = max_row_non_zero * m->indicies[i];
    uint64_t batch = i / m->nonZero;
    for (; result->values[index] != 0; index++)
      ;
    // Next free index

    result->values[index] = m->values[i];
    result->indicies[index] = batch;
  }
  free(occurrences);
  return result;
}

__global__ void fillELL(ELLMatrix *result, Matrix *matrix) {
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < matrix->cols) {
    uint64_t elements_added = col * result->nonZero;
    uint64_t added = 0;
    uint64_t dc = 0;
    for (uint64_t row = 0; row < result->rows; row++) {
      if (matrix->data[row * matrix->cols + col] != 0) {
        result->values[elements_added] = matrix->data[row * matrix->cols + col];
        result->indicies[elements_added] = row;
        elements_added++;
        added++;
        if (elements_added % result->nonZero == 0)
          break;
      } else {
        dc = row;
      }
    }
    for (; added < result->nonZero; added++) {
      result->values[elements_added] = 0;
      result->indicies[elements_added] = dc;
      elements_added++;
    }
  }
}
/**
 *
 * Converts a 2D Matrix into a ELLPACK struct and returns a pointer to the
 * struct
 *
 */
__host__ void toELL(Matrix *matrix, ELLMatrix *result) {

  result->rows = matrix->rows;
  result->cols = matrix->cols;

  uint64_t blockSize = 512;
  uint64_t numBlocks = (matrix->cols + blockSize - 1) / blockSize;

  ELLMatrix *d_result = toDevice(result);
  Matrix *d_matrix = toDevice(matrix);

  countNonZero<<<numBlocks, blockSize>>>(d_matrix, &d_result->nonZero);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error while calculating nonZero elements.");
    return;
  }

  err = cudaMemcpy(&result->nonZero, &d_result->nonZero, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error while copying nonZero from Device.");
    return;
  }

  __uint128_t size = (__uint128_t)result->nonZero * (__uint128_t)result->cols;

  if (size == 0) {
    result->indicies = NULL;
    result->values = NULL;
    return;
  }

  result->values = (float *)malloc(size * sizeof(float));
  if (!result->values) {
    free(result);
    return; // TODO: Error handling
  }
  result->indicies = (uint64_t *)malloc(size * sizeof(uint64_t));
  if (!result->indicies) {
    free(result->values);
    free(result);
    return; // TODO: Error handling
  }

  fillELL<<<numBlocks, blockSize>>>(d_result, d_matrix);
  cudaMemcpy(result->indicies, d_result->indicies,
             result->cols * result->nonZero * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(result->values, d_result->values,
             result->cols * result->nonZero * sizeof(float),
             cudaMemcpyDeviceToHost);
  freeFromDevice(d_result);
  freeFromDevice(d_matrix);

  return;
}

__global__ void fromELL(ELLMatrix *m, Matrix *result) {
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  uint64_t totalNonZero = m->cols * m->nonZero;

  if (idx < totalNonZero) {
    uint64_t col = idx / m->nonZero;

    float value = m->values[idx];
    uint64_t row = m->indicies[idx];

    if (value != 0) {
      result->data[row * m->cols + col] = value;
    }
  }
}

float ellEquals(ELLMatrix *a, ELLMatrix *b) {
  if (a->cols != b->cols || a->rows != b->rows || a->nonZero != b->nonZero) {
    return -1.0F;
  }
  float max = 0.0F;
  for (uint64_t i = 0; i < a->cols * a->nonZero; i++) {
    if (a->indicies[i] != b->indicies[i])
      return -2.0F;
    if (a->values[i] != b->values[i]) {
      float val = a->values[i] - b->values[i];
      val = val < 0 ? -val : val;
      max = val > max ? val : max;
    }
  }
  return max;
}

void printEll(ELLMatrix *m) {
  printf("ELL: \n");
  printf("cols: %lu \n", m->cols);
  printf("rows: %lu \n", m->rows);
  printf("nonZero: %lu \n", m->nonZero);
  for (int i = 0; i < m->cols * m->nonZero; i++) {
    printf("%f,", m->values[i]);
  }
  printf("\n");
  for (int i = 0; i < m->cols * m->nonZero; i++) {
    printf("%lu,", m->indicies[i]);
  }
}

__host__ void freeELL(ELLMatrix *ell) {
  free(ell->indicies);
  free(ell->values);
  free(ell);
}

__host__ ELLMatrix *generateRandomELL(uint64_t nonZero, uint64_t rows,
                                      uint64_t cols, float dcRate) {
  ELLMatrix *result = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  if (!result) {
    fprintf(stderr, "Could not allocate Memory!");
    return NULL;
  }
  createELL(rows, cols, nonZero, result);
  if (!result) {
    return NULL;
  }
  for (uint64_t i = 0; i < cols; i++) {
    uint64_t dc = 0;
    if ((float)rand() / (float)RAND_MAX < dcRate) {
      dc = (rand() % nonZero);
    }
    for (uint64_t j = 0; j < nonZero - dc; j++) {
      if (j >= cols * nonZero) {
        printf(" ");
      }
      result->values[i * nonZero + j] =
          (1000.0F * ((float)rand() / (float)RAND_MAX)) +
          ((float)rand() / (float)RAND_MAX);
      result->indicies[i * nonZero + j] = j;
    }
  }
  return result;
}
