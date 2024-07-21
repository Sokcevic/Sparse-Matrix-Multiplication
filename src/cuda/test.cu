#include "ELL.cuh"
#include "main.cu"
#include <bits/time.h>
#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
__host__ Matrix *generateRandomMatrix(float zeroRate, uint64_t rows,
                                      uint64_t cols) {
  Matrix *result = createMatrix(rows, cols);
  if (result == NULL) {
    exit(-2);
  }
  for (uint64_t i = 0; i < rows * cols; i++) {
    if ((float)rand() / (float)RAND_MAX < zeroRate) {
      result->data[i] = 0.0;
    } else {
      result->data[i] = (1000.0F * ((float)rand() / (float)RAND_MAX)) +
                        ((float)rand() / (float)RAND_MAX);
    }
  }
  return result;
}
__host__ ELLMatrix *generateRandomELL(uint64_t nonZero, uint64_t rows,
                                      uint64_t cols, float dcRate) {
  ELLMatrix *result = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  if (!result) {
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

void setup() { srand(69); }

void test_matmul(uint64_t rows_a, uint64_t cols_a, uint64_t rows_b,
                 uint64_t cols_b, float sparsity,
                 void (*f)(const void *, const void *, void *)) {
  Matrix *a;
  Matrix *b;
  Matrix *c;
  int incorrect = 0;
  float maxdiff = 0.0;
  int iterations = 100;
  for (int i = 0; i < iterations; i++) {
    a = generateRandomMatrix(sparsity, rows_a, cols_a);
    b = generateRandomMatrix(sparsity, rows_b, cols_b);
    c = createMatrix(rows_a, cols_b);

    ELLMatrix *al = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
    toELL(a, al);
    ELLMatrix *bl = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
    toELL(b, bl);
    ELLMatrix *cl = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
    toELL(c, cl);

    f(al, bl, cl);

    freeELL(al);
    freeELL(bl);

    Matrix *c2 = createMatrix(rows_a, cols_b);

    int numBlocks = 512;
    int blockSize = (cl->nonZero * cl->cols + numBlocks - 1) / numBlocks;
    ELLMatrix *d_cl = ELLtoDevice(cl);
    Matrix *d_c2 = toDevice(c2);

    fromELL<<<blockSize, numBlocks>>>(d_cl, d_c2);

    freeFromDevice(d_cl);
    freeELL(cl);

    blockSize = (a->cols * b->rows + numBlocks - 1) / numBlocks;
    matmul<<<blockSize, numBlocks>>>(a, b, c);
    freeMatrix(a);
    freeMatrix(b);

    float diff = matrixEqual(c, c2);
    freeMatrix(c);
    freeMatrix(c2);

    if (diff != 0) {
      incorrect++;
      maxdiff = maxdiff > diff ? maxdiff : diff;
    }
  }
  printf("%d/%d correct.\n", iterations - incorrect, iterations);
  printf("Max absolute difference: %f", maxdiff);
}

void testELLMul(uint64_t rows_a, uint64_t cols_a, uint64_t rows_b,
                uint64_t cols_b, float sparsity,
                void (*f)(const void *, const void *, void *)) {

  int iterations = 10;
  double times = 0;
  for (int i = 0; i < iterations; i++) {

    ELLMatrix *al = generateRandomELL(5, rows_a, cols_a, sparsity);
    ELLMatrix *bl = generateRandomELL(5, rows_b, cols_b, sparsity);
    ELLMatrix *cl = createELL(rows_a, cols_b, 0);

    printEll(al);
    printEll(bl);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    f(al, bl, cl);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double took =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    times += took;

    printf("took: %f\n", took);

    freeELL(al);
    freeELL(bl);
  }
  printf("avg time: %f", ((double)times) / ((double)iterations));
}

void testELL() {
  Matrix *a;
  Matrix *b;
  Matrix *c;
  int incorrect = 0;
  float maxdiff = 0.0;
  int iterations = 20;
  for (int i = 0; i < iterations; i++) {
    a = generateRandomMatrix(0.6, 300, 350);

    ELLMatrix *al = toELL(a);
    ELLMatrix *al_t = transpose(al);
    ELLMatrix *al_tt = transpose(al_t);
     Matrix *am = fromELL(al_tt);
    freeELL(al_tt);
    freeELL(al_t);
     float f = matrixEqual(am, a);
    printf("%f\n", f);

    freeMatrix(am);
    freeMatrix(a);
    freeELL(al);
  }
}

void test_matmul_large() {
  // test_matmul(300, 500, 500, 300, 0.3, &matr_mult_ellpack_v6);
}

__host__ Matrix *matmul(Matrix *a, Matrix *b) {
  Matrix *c = createMatrix(a->rows, b->cols);
  for (uint64_t i = 0; i < a->rows; ++i) {
    for (uint64_t j = 0; j < b->cols; ++j) {
      for (uint64_t k = 0; k < a->cols; ++k) {
        c->data[i * c->cols + j] +=
            a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
    }
  }
  return c;
}
**/

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
  /**
  Matrix *a = generateRandomMatrix(0.1, 4000, 5000);
  Matrix *d_a = toDevice(a);

  Matrix *b = generateRandomMatrix(0.1, 5000, 2000);
  Matrix *d_b = toDevice(b);

  Matrix *c = createMatrix(4, 2);
  Matrix *d_c = toDevice(c);

  dim3 dimBlock(32, 32);
  dim3 dimGrid((c->cols + dimBlock.x - 1) / 16,
               (c->rows + dimBlock.y - 1) / 16);

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i = 0; i < 10; i++) {
    matmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "CudaError: %s", cudaGetErrorString(err));
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double sum =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Took: %fs", sum);
  copyFromDevice(d_c, c);
**/

  ELLMatrix *a = generateRandomELL(4, 100, 100, 0.1);
  ELLMatrix *b = generateRandomELL(3, 100, 100, 0.1);
  ELLMatrix *c = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  createELL(100, 100, 0, c);

  matr_mult_ellpack_v6(a, b, c);

  printEll(a);
  printf("\n");
  printEll(b);
  printf("\n");
  printEll(c);

  // float f = matrixEqual(c, matmul(a, b));
  // printf("diff: %f", f);
  //  test_matmul_large();
  //  testELLMul(30000, 50000, 50000, 30000, 0.6, &matr_mult_ellpack_v7);
  //   testELL();
  //  char input[] = ""
  //                "src/TestELLInput.txt";
  //  ELLMatrix *test = textFileToEll(input);
  //  printEll(test);
  //  char output[] = ""
  //                 "src/TestEllOutput.txt";
  //  EllToTextFile(test, output);
  //  freeELL(test);
  return EXIT_SUCCESS;
}
