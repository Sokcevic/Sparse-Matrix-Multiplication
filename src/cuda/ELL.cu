#include "ELL.cuh"
#include <cstddef>
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

__host__ void copyFromDevice(ELLMatrix *src, ELLMatrix *target) {
  cudaError_t err =
      cudaMemcpy(&target->rows, &src->rows, 1, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(&target->cols, &src->cols, 1, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(&target->nonZero, &src->nonZero, 1, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(&target->values, &src->values, src->cols * src->nonZero, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(&target->indicies, &src->indicies, src->cols * src->nonZero, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Couldn't copy from device: %s\n", cudaGetErrorString(err));
    return;
  }

}

__host__ ELLMatrix *ELLtoDevice(ELLMatrix *matrix) {
  ELLMatrix *d_matrix;
  cudaError_t cudaError = cudaMalloc((void **)&d_matrix, sizeof(ELLMatrix));
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't allocate memory for indices on the GPU: %s\n",
            cudaGetErrorString(cudaError));
    return NULL;
  }

  cudaError = cudaMalloc((void **)&d_matrix->values,
                         matrix->cols * matrix->nonZero * sizeof(float));

  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't allocate memory for indices on the GPU: %s\n",
            cudaGetErrorString(cudaError));
    cudaFree(d_matrix);
    return NULL;
  }

  cudaError = cudaMalloc((void **)&d_matrix->indicies,
                         matrix->cols * matrix->nonZero * sizeof(uint64_t));

  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't allocate memory for indices on the GPU: %s\n",
            cudaGetErrorString(cudaError));
    cudaFree(d_matrix->values);
    cudaFree(d_matrix);
    return NULL;
  }
  cudaMemcpy(d_matrix, matrix, 1, cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "Couldn't copy ELLMatrix to the GPU: %s\n",
            cudaGetErrorString(cudaError));
    cudaFree(d_matrix->indicies);
    cudaFree(d_matrix->values);
    cudaFree(d_matrix);
    return NULL;
  }
  return d_matrix;
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
      (uint64_t *)malloc(result->cols * result->nonZero * sizeof(uint64_t));
  if (!result->indicies) {
    free(result);
    return; // TODO: Error Handling
  }
  result->values =
      (float *)malloc(result->cols * result->nonZero * sizeof(float));
  if (!result->values) {
    free(result->indicies);
    free(result);
    return; // TODO: Error Handling
  }
  return;
}

__global__ void transpose(ELLMatrix *m) {
  uint64_t *occurrences = (uint64_t *)malloc(m->rows * sizeof(uint64_t));
  if (!occurrences) {
    return; // TODO: Error handling
  }

  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t max_row_non_zero = 0;

  // initialzie occurences
  for (uint64_t i = tid; i < m->rows; i += blockDim.x * gridDim.x) {
    occurrences[i] = 0;
  }
  __syncthreads();

  // calculate values per row
  for (uint64_t i = tid; i < m->cols * m->nonZero;
       i += blockIdx.x * gridDim.x) {
    if (m->values[i] != 0) {
      atomicAdd((unsigned long long *)occurrences[m->indicies[i]], 1);
    }
  }
  __syncthreads();

  // find max
  for (uint64_t i = tid; i < m->rows; i += blockIdx.x * gridDim.x) {
    atomicMax((unsigned long long *)&max_row_non_zero,
              (unsigned long long)occurrences[i]);
  }

  __syncthreads();

  // create new value and indices arrays
  size_t size = m->rows * max_row_non_zero;
  float *values = (float *)malloc(size * sizeof(float));
  if (!values) {
    return;
  }
  uint64_t *indicies = (uint64_t *)malloc(size * sizeof(uint64_t *));
  if (!indicies) {
    free(values);
    return;
  }
  // Initialize them
  for (uint64_t i = tid; i < size; i += blockDim.x * gridDim.x) {
    values[i] = 0;
    indicies[i] = 0;
  }
  __syncthreads();

  float zero = 0;

  for (uint64_t i = tid; i < m->cols * m->nonZero;
       i += blockDim.x * gridDim.x) {
    uint64_t index = max_row_non_zero * m->indicies[i];
    uint64_t batch = i / m->nonZero;
    while (atomicCAS((unsigned long long *)&values[index],
                     *(unsigned long long *)&zero,
                     *(unsigned long long *)&m->values[i]) != 0) {
      index++;
    }
    indicies[index] = batch;
  }
  __syncthreads();
  if (tid == 0) {
    free(m->indicies);
    free(m->values);
    m->indicies = indicies;
    m->values = values;
  }
  return;
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
      if (matrix->data[row][col] != 0) {
        count++;
      }
    }
    colCount[col] = count;
  }
  __syncthreads();
  atomicMax((unsigned long long *)nonZero, colCount[col]);
  return;
}

__global__ void fillELL(ELLMatrix *result, Matrix *matrix) {
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < matrix->cols) {
    uint64_t elements_added = col * result->nonZero;
    uint64_t added = 0;
    uint64_t dc = 0;
    for (uint64_t row = 0; row < result->rows; row++) {
      if (matrix->data[row][col] != 0) {
        result->values[elements_added] = matrix->data[row][col];
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

  int blockSize = 256;
  int numBlocks = (matrix->cols + blockSize - 1) / blockSize;

  ELLMatrix *d_result = ELLtoDevice(result);
  Matrix *d_matrix = toDevice(matrix);

  countNonZero<<<numBlocks, blockSize>>>(d_matrix, &d_result->nonZero);

  cudaMemcpy(&result->nonZero, &d_result->nonZero, 1, cudaMemcpyDeviceToHost);

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
             result->cols * result->nonZero, cudaMemcpyDeviceToHost);
  cudaMemcpy(result->values, d_result->values, result->cols * result->nonZero,
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
      result->data[row][col] = value;
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

