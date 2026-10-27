#include "ELL.h"
#include "matrix.h"
#include "multiply.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define THREADS_PER_BLOCK 256
#define TILE_SIZE 16

// Prints the error and returns false if err is not cudaSuccess
static bool cudaCheck(cudaError_t err, const char *action) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error while %s: %s\n", action,
            cudaGetErrorString(err));
    return false;
  }
  return true;
}

// a * b as size_t, false on overflow
static bool multiplySize(uint64_t a, uint64_t b, size_t elementSize,
                         size_t *out) {
  if (a != 0 && b > SIZE_MAX / elementSize / a) {
    fprintf(stderr, "Matrix is too large!\n");
    return false;
  }
  *out = a * b * elementSize;
  return true;
}

/**
 * Copies the arrays of host to the GPU. device is a copy of host whose
 * pointers point to GPU memory, so it can be passed to kernels by value.
 */
static bool ellToDevice(const ELLMatrix *host, ELLMatrix *device) {
  *device = *host;
  device->values = NULL;
  device->indicies = NULL;
  uint64_t size = host->cols * host->nonZero;
  if (size == 0) {
    return true;
  }
  return cudaCheck(cudaMalloc((void **)&device->values, size * sizeof(float)),
                   "allocating memory") &&
         cudaCheck(cudaMalloc((void **)&device->indicies,
                              size * sizeof(uint64_t)),
                   "allocating memory") &&
         cudaCheck(cudaMemcpy(device->values, host->values,
                              size * sizeof(float), cudaMemcpyHostToDevice),
                   "copying to the GPU") &&
         cudaCheck(cudaMemcpy(device->indicies, host->indicies,
                              size * sizeof(uint64_t), cudaMemcpyHostToDevice),
                   "copying to the GPU");
}

static void freeDeviceELL(ELLMatrix *device) {
  cudaFree(device->values);
  cudaFree(device->indicies);
  device->values = NULL;
  device->indicies = NULL;
}

/**
 * Dot product of column aCol of at (= row aCol of A) and column bCol of b.
 * Same merge as sparseDot in multiply.c.
 */
static __device__ float sparseDot(ELLMatrix at, uint64_t aCol, ELLMatrix b,
                                  uint64_t bCol) {
  uint64_t a_index = aCol * at.nonZero;
  uint64_t b_index = bCol * b.nonZero;
  uint64_t a_end = a_index + at.nonZero;
  uint64_t b_end = b_index + b.nonZero;
  float sum = 0;
  while (a_index < a_end && b_index < b_end && at.values[a_index] != 0 &&
         b.values[b_index] != 0) {
    uint64_t a_row = at.indicies[a_index];
    uint64_t b_row = b.indicies[b_index];
    if (a_row == b_row) {
      sum += at.values[a_index] * b.values[b_index];
      a_index++;
      b_index++;
    } else if (a_row < b_row) {
      a_index++;
    } else {
      b_index++;
    }
  }
  return sum;
}

// counts[i] = number of non-zero values in column i of the result
static __global__ void countNonZeroKernel(ELLMatrix at, ELLMatrix b,
                                          uint64_t *counts) {
  uint64_t col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= b.cols) {
    return;
  }
  uint64_t count = 0;
  for (uint64_t row = 0; row < at.cols; row++) {
    if (sparseDot(at, row, b, col) != 0) {
      count++;
    }
  }
  counts[col] = count;
}

// writes column col of the result; result has to be zero-initialized
static __global__ void ellMulKernel(ELLMatrix at, ELLMatrix b,
                                    ELLMatrix result) {
  uint64_t col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= b.cols) {
    return;
  }
  uint64_t result_index = col * result.nonZero;
  for (uint64_t row = 0; row < at.cols; row++) {
    float sum = sparseDot(at, row, b, col);
    if (sum != 0) {
      result.values[result_index] = sum;
      result.indicies[result_index] = row;
      result_index++;
    }
  }
}

/**
 * Computes *res = A * B with at = transpose(A). *res has to be NULL and is
 * only set on success.
 */
static bool ellMulOnDevice(const ELLMatrix *at, const ELLMatrix *b,
                           ELLMatrix **res) {
  bool ok = false;
  ELLMatrix d_at = {}, d_b = {}, d_res = {};
  uint64_t *d_counts = NULL;
  uint64_t *counts = NULL;
  uint64_t maxNonZero = 0;
  uint64_t blocks = (b->cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  size_t resultSize;

  if (!ellToDevice(at, &d_at) || !ellToDevice(b, &d_b) ||
      !cudaCheck(cudaMalloc((void **)&d_counts, b->cols * sizeof(uint64_t)),
                 "allocating memory")) {
    goto cleanup;
  }

  // nonZero of the result: maximum number of non-zero values in a column
  countNonZeroKernel<<<blocks, THREADS_PER_BLOCK>>>(d_at, d_b, d_counts);
  if (!cudaCheck(cudaGetLastError(), "counting non-zero values")) {
    goto cleanup;
  }
  counts = (uint64_t *)malloc(b->cols * sizeof(uint64_t));
  if (!counts) {
    fprintf(stderr, "Memory allocation failed!\n");
    goto cleanup;
  }
  if (!cudaCheck(cudaMemcpy(counts, d_counts, b->cols * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost),
                 "counting non-zero values")) {
    goto cleanup;
  }
  for (uint64_t i = 0; i < b->cols; i++) {
    maxNonZero = counts[i] > maxNonZero ? counts[i] : maxNonZero;
  }

  *res = createELL(at->cols, b->cols, maxNonZero);
  if (!*res) {
    goto cleanup;
  }
  if (maxNonZero == 0) {
    ok = true;
    goto cleanup;
  }

  // calculate the result, padding stays 0 from createELL's calloc
  resultSize = b->cols * maxNonZero;
  if (!ellToDevice(*res, &d_res)) {
    goto cleanup;
  }
  ellMulKernel<<<blocks, THREADS_PER_BLOCK>>>(d_at, d_b, d_res);
  ok = cudaCheck(cudaGetLastError(), "multiplying") &&
       cudaCheck(cudaMemcpy((*res)->values, d_res.values,
                            resultSize * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copying the result") &&
       cudaCheck(cudaMemcpy((*res)->indicies, d_res.indicies,
                            resultSize * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost),
                 "copying the result");

cleanup:
  if (!ok) {
    freeELL(*res);
    *res = NULL;
  }
  free(counts);
  cudaFree(d_counts);
  freeDeviceELL(&d_at);
  freeDeviceELL(&d_b);
  freeDeviceELL(&d_res);
  return ok;
}

// Ellpack on the GPU
extern "C" void matr_mult_ellpack_V3(const void *a, const void *b, void *result) {
  ELLMatrix *at = transpose((const ELLMatrix *)a); // read through rows of a
  if (!at) {
    return;
  }
  ELLMatrix *res = NULL;
  if (ellMulOnDevice(at, (const ELLMatrix *)b, &res)) {
    *((ELLMatrix *)result) = *res;
    free(res);
  }
  freeELL(at);
}

// c = a * b for dense row-major matrices a (n x m) and b (m x p)
static __global__ void denseMulKernel(const float *a, const float *b, float *c,
                                      uint64_t n, uint64_t m, uint64_t p) {
  uint64_t row = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < p) {
    float sum = 0;
    for (uint64_t k = 0; k < m; k++) {
      sum += a[row * m + k] * b[k * p + col];
    }
    c[row * p + col] = sum;
  }
}

// row-major dense copy of m, NULL on failure
static float *ellToDense(const ELLMatrix *m) {
  size_t size;
  if (!multiplySize(m->rows, m->cols, sizeof(float), &size)) {
    return NULL;
  }
  float *dense = (float *)calloc(size, 1);
  if (!dense) {
    fprintf(stderr, "Memory allocation failed!\n");
    return NULL;
  }
  for (uint64_t i = 0; i < m->cols * m->nonZero; i++) {
    if (m->values[i] != 0) {
      dense[m->indicies[i] * m->cols + i / m->nonZero] = m->values[i];
    }
  }
  return dense;
}

// copies the dense row-major matrix to the GPU, NULL on failure
static float *denseToDevice(const float *host, size_t size) {
  float *device = NULL;
  if (!cudaCheck(cudaMalloc((void **)&device, size), "allocating memory")) {
    return NULL;
  }
  if (host && !cudaCheck(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice),
                         "copying to the GPU")) {
    cudaFree(device);
    return NULL;
  }
  return device;
}

// normale MatMul on the GPU
extern "C" void matr_mult_ellpack_V4(const void *a, const void *b, void *result) {
  const ELLMatrix *matrix_a = (const ELLMatrix *)a;
  const ELLMatrix *matrix_b = (const ELLMatrix *)b;
  uint64_t n = matrix_a->rows;
  uint64_t m = matrix_a->cols;
  uint64_t p = matrix_b->cols;

  float *ha = ellToDense(matrix_a);
  float *hb = ellToDense(matrix_b);
  float *hc = NULL;
  float **rows = NULL;
  float *da = NULL, *db = NULL, *dc = NULL;
  size_t cSize;
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  dim3 dimGrid((unsigned)((p + TILE_SIZE - 1) / TILE_SIZE),
               (unsigned)((n + TILE_SIZE - 1) / TILE_SIZE));

  if (!ha || !hb || !multiplySize(n, p, sizeof(float), &cSize)) {
    goto cleanup;
  }
  hc = (float *)malloc(cSize);
  rows = (float **)malloc(n * sizeof(float *));
  if (!hc || !rows) {
    fprintf(stderr, "Memory allocation failed!\n");
    goto cleanup;
  }
  da = denseToDevice(ha, n * m * sizeof(float));
  db = da ? denseToDevice(hb, m * p * sizeof(float)) : NULL;
  dc = db ? denseToDevice(NULL, cSize) : NULL;
  if (!dc) {
    goto cleanup;
  }

  denseMulKernel<<<dimGrid, dimBlock>>>(da, db, dc, n, m, p);
  if (!cudaCheck(cudaGetLastError(), "multiplying") ||
      !cudaCheck(cudaMemcpy(hc, dc, cSize, cudaMemcpyDeviceToHost),
                 "copying the result")) {
    goto cleanup;
  }

  { // view hc as Matrix to convert it with toELL
    for (uint64_t i = 0; i < n; i++) {
      rows[i] = &hc[i * p];
    }
    Matrix mc = {n, p, rows};
    ELLMatrix *res = toELL(&mc);
    if (res) {
      *((ELLMatrix *)result) = *res;
      free(res);
    }
  }

cleanup:
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  free(ha);
  free(hb);
  free(hc);
  free(rows);
}
