#include "ELL.cuh"
#include "matrix.cuh"
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
  ELLMatrix *m = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  if (!m) {
    fprintf(stderr, "Error allocating memory!");
    return;
  }
  cudaError_t err =
      cudaMemcpy(m, d_matrix, sizeof(ELLMatrix), cudaMemcpyDeviceToHost);
  cudaFree(m->indicies);
  cudaFree(m->values);
  cudaFree(d_matrix);
  free(m);
}
__host__ ELLMatrix *createELL(uint64_t rows, uint64_t cols, uint64_t nonZero) {
  ELLMatrix *result = (ELLMatrix *)malloc(sizeof(ELLMatrix));
  if (!result) {
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->rows = rows;
  result->cols = cols;
  result->nonZero = nonZero;
  if (nonZero == 0) {
    result->indicies = NULL;
    result->values = NULL;
    return result;
  }
  result->indicies =
      (uint64_t *)calloc(result->cols * result->nonZero, sizeof(uint64_t));
  if (!result->indicies) {
    free(result);
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->values =
      (float *)calloc(result->cols * result->nonZero, sizeof(float));
  if (!result->values) {
    free(result->indicies);
    free(result);
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  return result;
}

__host__ ELLMatrix *transpose(ELLMatrix *m) {
  uint64_t *occurrences = NULL;
  if (m->nonZero != 0) {
    occurrences = (uint64_t *)calloc(m->rows, sizeof(uint64_t));
    if (!occurrences) {
      fprintf(stderr, "Memory allocation failed!\n");
      return NULL;
    }
  }
  // calculate nonZero of transposed matrix
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
  ELLMatrix *result = createELL(m->cols, m->rows, max_row_non_zero);
  if (!result) {
    if (occurrences)
      free(occurrences);
    return NULL;
  }
  // write values and indices to transposed matrix
  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    uint64_t index = max_row_non_zero * m->indicies[i];
    uint64_t batch = i / m->nonZero;
    for (; result->values[index] != 0; index++)
      ;
    // Next free index

    result->values[index] = m->values[i];
    result->indicies[index] = batch;
  }
  if (occurrences != NULL)
    free(occurrences);
  return result;
}

/**
 *
 * Converts a 2D Matrix into a ELLPACK struct and returns a pointer to the
 * struct
 *
 */
ELLMatrix *toELL(Matrix *matrix) {

  ELLMatrix *result; // initialize
  uint64_t m = matrix->rows;
  uint64_t n = matrix->cols;

  result = (ELLMatrix *)malloc(sizeof(ELLMatrix));
  if (!result) {
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->rows = m;
  result->cols = n;

  uint64_t nonZero[n];
  memset(nonZero, 0, sizeof(nonZero));
  // calculate max number of non Zero values per column
  for (uint64_t i = 0; i < m; i++) // Use Cache properly
    for (uint64_t j = 0; j < n; j++)
      if (matrix->data[i * matrix->cols + j] != 0)
        nonZero[j]++;

  uint64_t max = 0;
  for (uint64_t i = 0; i < n; i++)
    if (nonZero[i] > max)
      max = nonZero[i];

  result->nonZero = max;
  __uint128_t size = (__uint128_t)max * (__uint128_t)n;

  if (size == 0) {
    result->indicies = NULL;
    result->values = NULL;
    return result;
  }

  result->values = (float *)calloc(size, sizeof(float));
  if (!result->values) {
    free(result);
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  result->indicies = (uint64_t *)calloc(size, sizeof(uint64_t));
  if (!result->indicies) {
    free(result->values);
    free(result);
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }

  uint64_t elements_added = 0;
  for (uint64_t i = 0; i < n; i++) { // iterate columns
    uint64_t added = 0;
    uint64_t dc = 0;
    for (uint64_t j = 0; j < m; j++) { // iterate rows
      if (matrix->data[j * matrix->cols + i] != 0) {
        (result->values)[elements_added] = matrix->data[j * matrix->cols + i];
        (result->indicies)[elements_added] = j;
        elements_added++;
        added++;
        if (elements_added % (max) == 0)
          break;
      } else {
        dc = j;
      }
    }
    for (; added < max; added++) {
      (result->values)[elements_added] = 0;
      (result->indicies)[elements_added] = dc;
      elements_added++;
    }
  }

  return result;
}

__host__ Matrix *fromELL(ELLMatrix *m) {
  Matrix *result = createMatrix(m->rows, m->cols);

  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    if (m->values[i] != 0) {
      result->data[m->cols * m->indicies[i] + (i / m->nonZero)] = m->values[i];
    }
  }
  return result;
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
  ELLMatrix *result = (ELLMatrix *)createELL(rows, cols, nonZero);
  if (!result) {
    return NULL;
  }
  if (nonZero == 0) {
    return result;
  }
  for (uint64_t i = 0; i < cols; i++) {
    uint64_t dc = 0;
    if ((float)rand() / (float)RAND_MAX < dcRate) {
      dc = (rand() % nonZero);
    }
    for (uint64_t j = 0; j < nonZero - dc; j++) {
      if (j >= cols * nonZero) {
        break;
      }
      result->values[i * nonZero + j] =
          (1000.0F * ((float)rand() / (float)RAND_MAX)) +
          ((float)rand() / (float)RAND_MAX);
      result->indicies[i * nonZero + j] = j;
    }
  }
  return result;
}
