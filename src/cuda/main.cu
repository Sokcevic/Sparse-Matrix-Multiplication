#include "ELL.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <stdlib.h>
#include <tmmintrin.h>

__global__ void maxNonZeroMul(ELLMatrix *a, ELLMatrix *b, uint64_t *result) {
  extern __shared__ uint64_t nonZero[];

  uint64_t b_batch = blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t a_batch = blockDim.y * blockIdx.y + threadIdx.y;

  if (threadIdx.y == 0) {
    nonZero[threadIdx.x] = 0;
  }
  __syncthreads();

  if (b_batch < b->cols && a_batch < a->cols) {
    float sum = 0;
    for (uint64_t i = 0, j = 0; i < b->nonZero && j < a->nonZero;) {
      if (b->indicies[b_batch * b->nonZero + i] ==
          a->indicies[a_batch * a->nonZero + j]) {
        sum += b->values[b_batch * b->nonZero + i] *
               a->values[a_batch * a->nonZero + j];
        i++;
        j++;
      } else if (b->indicies[b_batch * b->nonZero + i] <
                 a->indicies[a_batch * a->nonZero + j]) {
        i++;
      } else {
        j++;
      }
    }
    if (sum != 0) {
      atomicAdd((unsigned long long *)&nonZero[threadIdx.x], 1);
    }
  }
  __syncthreads();

  if (threadIdx.y == 0) {
    atomicMax((unsigned long long *)result,
              (unsigned long long)nonZero[threadIdx.x]);
  }
  __syncthreads();
  atomicMax((unsigned long long *)result,
            (unsigned long long)nonZero[b_batch]);
}

__global__ void ELLMul(ELLMatrix *a, ELLMatrix *b, ELLMatrix *c) {
    uint64_t row = blockDim.y * blockIdx.y + threadIdx.y;
    uint64_t col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < a->rows && col < b->cols) {
        float sum = 0.0;
        for (uint64_t i = 0, j = 0; i < a->nonZero && j < b->nonZero;) {
            uint64_t a_index = a->indicies[row * a->nonZero + i];
            uint64_t b_index = b->indicies[col * b->nonZero + j];

            if (a_index == b_index) {
                sum += a->values[row * a->nonZero + i] * b->values[col * b->nonZero + j];
                i++;
                j++;
            } else if (a_index < b_index) {
                i++;
            } else {
                j++;
            }
        }

        if (sum != 0.0) {
            uint64_t idx = row * b->cols + col;
            c->indicies[idx] = row; 
            c->values[idx] = sum;
        }
    }
}
__host__ void matr_mult_ellpack_v6(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = transpose((ELLMatrix *)a);
  if (!matrix_a) {
    fprintf(stderr, "Could not allocate memory!");
    return;
  }

  ELLMatrix *matrix_b = (ELLMatrix *)b;

  // Check if multiplication is possible
  if (matrix_a->rows != matrix_b->rows) {
    return; // Not allowed TODO: Error Handling
  }
  cudaDeviceSynchronize();

  ELLMatrix *d_a = toDevice(matrix_a);
  ELLMatrix *d_b = toDevice(matrix_b);
  uint64_t *d_nonZero;
  cudaMalloc((void **)&d_nonZero, sizeof(uint64_t));

  dim3 dimBlock(32, 32);
  dim3 dimGrid((matrix_b->cols + dimBlock.x - 1) / dimBlock.x,
               (matrix_a->cols + dimBlock.y - 1) / dimBlock.y);

  maxNonZeroMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_nonZero);
  uint64_t maxNonZero;
  cudaMemcpy(&maxNonZero, d_nonZero, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_nonZero);

  ELLMatrix *res = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  if (!res) {
    freeELL(matrix_a);
    return;
  }

  createELL(matrix_a->cols, matrix_b->cols, maxNonZero, res);

  ELLMatrix *d_res = toDevice(res);
  ELLMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_res);
  copyFromDevice(d_res, res);

  *((ELLMatrix *)result) = *res;

  free(res);
  freeELL(matrix_a);
}

int main(void) {

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    return 1;
  }

  if (deviceCount == 0) {
    printf("There are no available CUDA devices.\n");
    return 1;
  } else {
    printf("Detected %d CUDA-capable device(s).\n", deviceCount);
  }

  ELLMatrix *a = generateRandomELL(4, 100000000, 100000000, 0.9);
  ELLMatrix *b = generateRandomELL(3, 100000000, 100000000, 0.9);
  ELLMatrix *c = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  createELL(100, 100, 0, c);

  matr_mult_ellpack_v6(a, b, c);

  printEll(a);
  printf("\n");
  printEll(b);
  printf("\n");
  printEll(c);

  return EXIT_SUCCESS;
}
