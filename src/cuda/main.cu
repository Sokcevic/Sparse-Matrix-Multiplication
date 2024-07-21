#include "ELL.cuh"
#include "Parser.cuh"
#include "main.cuh"
#include "matrix.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <getopt.h>
#include <stdint.h>
#include <stdlib.h>
#include <tmmintrin.h>

// Ellpack parallel
void matr_mult_ellpack(const void *a, const void *b, void *result) {

  // Cast to structure
  ELLMatrix *matrix_a = transpose((ELLMatrix *)a);
  ELLMatrix *matrix_b = (ELLMatrix *)b;

  if (matrix_a == NULL) {
    return;
  }
  if (matrix_a->nonZero == 0 || matrix_b->nonZero == 0) {
    ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, 0);
    freeELL(matrix_a);
    if (res) {
      *((ELLMatrix *)result) = *res;
      free(res);
    }
    return;
  }

  uint64_t maxNonZero = 0;
#pragma omp parallel for
  for (uint64_t i = 0; i < matrix_b->cols; i++) {
    uint64_t nonZero = 0;
#pragma omp parallel for
    for (uint64_t j = 0; j < matrix_a->cols; j++) {
      uint64_t a_batch_index = j * matrix_a->nonZero;
      uint64_t b_batch_index = i * matrix_b->nonZero;
      float sum = 0;
      while (a_batch_index < (j + 1) * matrix_a->nonZero &&
             b_batch_index < (i + 1) * matrix_b->nonZero) {
        if (matrix_a->values[a_batch_index] == 0 ||
            matrix_b->values[b_batch_index] == 0) {
          break;
        }
        uint64_t a_next_index = matrix_a->indicies[a_batch_index];
        uint64_t b_next_index = matrix_b->indicies[b_batch_index];
        if (a_next_index == b_next_index) {
          sum +=
              matrix_a->values[a_batch_index] * matrix_b->values[b_batch_index];
          a_batch_index++;
          b_batch_index++;
        } else if (a_next_index < b_next_index) {
          a_batch_index++;
        } else {
          b_batch_index++;
        }
      }
      if (sum != 0) {
#pragma omp atomic
        nonZero++;
      }
    }
#pragma omp atomic
    maxNonZero = maxNonZero > nonZero ? maxNonZero : nonZero;
  }

  ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, maxNonZero);
  if (!res) {
    freeELL(matrix_a);
    return;
  }

#pragma omp parallel for
  for (uint64_t i = 0; i < matrix_b->cols; i++) {
    uint64_t result_index = i * res->nonZero;
    for (uint64_t j = 0; j < matrix_a->cols; j++) {
      uint64_t a_batch_index = j * matrix_a->nonZero;
      uint64_t b_batch_index = i * matrix_b->nonZero;
      float sum = 0;
      while (a_batch_index < (j + 1) * matrix_a->nonZero &&
             b_batch_index < (i + 1) * matrix_b->nonZero) {
        if (matrix_a->values[a_batch_index] == 0 ||
            matrix_b->values[b_batch_index] == 0) {
          break;
        }
        uint64_t a_next_index = matrix_a->indicies[a_batch_index];
        uint64_t b_next_index = matrix_b->indicies[b_batch_index];
        if (a_next_index == b_next_index) {
          sum +=
              matrix_a->values[a_batch_index] * matrix_b->values[b_batch_index];
          a_batch_index++;
          b_batch_index++;
        } else if (a_next_index < b_next_index) {
          a_batch_index++;
        } else {
          b_batch_index++;
        }
      }
      if (sum != 0) {
        res->values[result_index] = sum;
        res->indicies[result_index] = j;
        result_index++;
      }
    }
  }

  *((ELLMatrix *)result) = *res;

  free(res);
  freeELL(matrix_a);
}

// Ellpack normal
void matr_mult_ellpack_V1(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = transpose((ELLMatrix *)a);
  ELLMatrix *matrix_b = (ELLMatrix *)b;

  if (matrix_a == NULL) {
    return;
  }

  if (matrix_a->nonZero == 0 || matrix_b->nonZero == 0) {
    ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, 0);
    freeELL(matrix_a);
    if (res) {
      *((ELLMatrix *)result) = *res;
      free(res);
    }
    return;
  }

  uint64_t maxNonZero = 0;
  for (uint64_t i = 0; i < matrix_b->cols; i++) {
    uint64_t nonZero = 0;
    for (uint64_t j = 0; j < matrix_a->cols; j++) {
      uint64_t a_batch_index = j * matrix_a->nonZero;
      uint64_t b_batch_index = i * matrix_b->nonZero;
      float sum = 0;
      while (a_batch_index < (j + 1) * matrix_a->nonZero &&
             b_batch_index < (i + 1) * matrix_b->nonZero) {
        if (matrix_a->values[a_batch_index] == 0 ||
            matrix_b->values[b_batch_index] == 0) {
          break;
        }
        uint64_t a_next_index = matrix_a->indicies[a_batch_index];
        uint64_t b_next_index = matrix_b->indicies[b_batch_index];
        if (a_next_index == b_next_index) {
          sum +=
              matrix_a->values[a_batch_index] * matrix_b->values[b_batch_index];
          a_batch_index++;
          b_batch_index++;
        } else if (a_next_index < b_next_index) {
          a_batch_index++;
        } else {
          b_batch_index++;
        }
      }
      if (sum != 0) {
        nonZero++;
      }
    }
    maxNonZero = maxNonZero > nonZero ? maxNonZero : nonZero;
  }

  ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, maxNonZero);
  if (!res) {
    freeELL(matrix_a);
    return;
  }

  uint64_t result_index = 0;
  for (uint64_t i = 0; i < matrix_b->cols; i++) {
    result_index = i * res->nonZero;
    for (uint64_t j = 0; j < matrix_a->cols; j++) {
      uint64_t a_batch_index = j * matrix_a->nonZero;
      uint64_t b_batch_index = i * matrix_b->nonZero;
      float sum = 0;
      while (a_batch_index < (j + 1) * matrix_a->nonZero &&
             b_batch_index < (i + 1) * matrix_b->nonZero) {
        if (matrix_a->values[a_batch_index] == 0 ||
            matrix_b->values[b_batch_index] == 0) {
          break;
        }
        uint64_t a_next_index = matrix_a->indicies[a_batch_index];
        uint64_t b_next_index = matrix_b->indicies[b_batch_index];
        if (a_next_index == b_next_index) {
          sum +=
              matrix_a->values[a_batch_index] * matrix_b->values[b_batch_index];
          a_batch_index++;
          b_batch_index++;
        } else if (a_next_index < b_next_index) {
          a_batch_index++;
        } else {
          b_batch_index++;
        }
      }
      if (sum != 0) {
        res->values[result_index] = sum;
        res->indicies[result_index] = j;
        result_index++;
      }
    }
  }

  *((ELLMatrix *)result) = *res;

  free(res);
  freeELL(matrix_a);
}

// normale MatMul SIMD
void matr_mult_ellpack_V2(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = (ELLMatrix *)a;
  ELLMatrix *matrix_b = transpose((ELLMatrix *)b);

  if (matrix_b == NULL) {
    return;
  }

  if (matrix_a->nonZero == 0 || matrix_b->nonZero == 0) {
    ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, 0);
    freeELL(matrix_b);
    if (res) {
      *((ELLMatrix *)result) = *res;
      free(res);
    }
    return;
  }

  // initialize result
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  Matrix *mr = createMatrix(ma->rows, mb->rows);

  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->rows; j++) {
      if (matrix_a->cols > 3) {
        __m128 sum = _mm_setzero_ps(); // Initialize sum to zero
        for (uint64_t k = 0; k < matrix_a->cols - (matrix_a->cols % 4);
             k += 4) {
          __m128 a_values = _mm_loadu_ps(&ma->data[i * ma->cols + k]);
          __m128 b_values = _mm_loadu_ps(&mb->data[j * mb->cols + k]);
          __m128 product = _mm_mul_ps(a_values, b_values);
          sum = _mm_add_ps(sum, product);
        }
        float temp[4];
        _mm_storeu_ps(temp, sum);

        mr->data[i * mr->cols + j] += temp[0] + temp[1] + temp[2] + temp[3];
      }
      for (int k = matrix_a->cols - (matrix_a->cols % 4); k < matrix_a->cols;
           k++) {
        mr->data[i * mr->cols + j] +=
            ma->data[i * ma->cols + k] * mb->data[j * mb->cols + k];
      }
    }
  }
  ELLMatrix *mx = toELL(mr);
  *((ELLMatrix *)result) = *mx;

  free(mx);
  freeMatrix(ma);
  freeMatrix(mb);
  freeELL(matrix_b);
  freeMatrix(mr);
}

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
  atomicMax((unsigned long long *)result, (unsigned long long)nonZero[b_batch]);
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
        sum +=
            a->values[row * a->nonZero + i] * b->values[col * b->nonZero + j];
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
__host__ void matr_mult_ellpack_V3(const void *a, const void *b, void *result) {
  // Cast to structure
  ELLMatrix *matrix_a = transpose((ELLMatrix *)a);
  cudaDeviceSynchronize();
  if (!matrix_a) {
    fprintf(stderr, "Could not allocate memory!");
    return;
  }

  ELLMatrix *matrix_b = (ELLMatrix *)b;

  // Check if multiplication is possible
  if (matrix_a->rows != matrix_b->rows) {
    fprintf(stderr, "Matrices have the wrong dimension!");
    return;
  }

  ELLMatrix *d_a = toDevice(matrix_a);
  ELLMatrix *d_b = toDevice(matrix_b);
  if (!d_a || !d_b) {
    fprintf(stderr, "Could not allocate memory on device!");
    return;
  }
  uint64_t *d_nonZero;
  cudaError_t err = cudaMalloc((void **)&d_nonZero, sizeof(uint64_t));
  if (err != cudaSuccess) {
    fprintf(stderr, "Encountered a cuda error: %s", cudaGetErrorString(err));
    return;
  }

  dim3 dimBlock(16, 16);
  dim3 dimGrid((matrix_b->cols + dimBlock.x - 1) / dimBlock.x,
               (matrix_a->cols + dimBlock.y - 1) / dimBlock.y);

  maxNonZeroMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_nonZero);
  uint64_t maxNonZero;
  err = cudaMemcpy(&maxNonZero, d_nonZero, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Encountered a cuda error: %s", cudaGetErrorString(err));
    return;
  }

  err = cudaFree(d_nonZero);
  if (err != cudaSuccess) {
    fprintf(stderr, "Encountered a cuda error: %s", cudaGetErrorString(err));
    return;
  }

  ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, maxNonZero);
  if (!res) {
    return;
  }

  ELLMatrix *d_res = toDevice(res);
  if (!d_res) {
    return;
  }
  ELLMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_res);
  copyFromDevice(d_res, res);

  *((ELLMatrix *)result) = *res;

  free(res);
  freeELL(matrix_a);
  freeELL(matrix_b);
  cudaFree(d_nonZero);
  freeFromDevice(d_res);
  freeFromDevice(d_a);
  freeFromDevice(d_b);
}

__host__ void matr_mult_ellpack_V4(const void *a, const void *b, void *result) {
  Matrix *matrix_a = fromELL((ELLMatrix *)a);
  Matrix *matrix_b = fromELL((ELLMatrix *)b);
  Matrix *c = createMatrix(matrix_a->rows, matrix_b->cols);

  Matrix *d_a = toDevice(matrix_a);
  Matrix *d_b = toDevice(matrix_b);
  Matrix *d_c = toDevice(c);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((matrix_b->cols + dimBlock.x - 1) / dimBlock.x,
               (matrix_a->cols + dimBlock.y - 1) / dimBlock.y);
  matmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

  copyFromDevice(d_c, c);

  freeMatrix(matrix_a);
  freeMatrix(matrix_b);
  freeFromDevice(d_c);
  freeFromDevice(d_b);
  freeFromDevice(d_a);
  ELLMatrix *c_ell = toELL(c);
  *((ELLMatrix *)result) = *c_ell;
  freeELL(c_ell);
  freeMatrix(c);
}

void printHelpMessage() {
  printf("Help Message\n"
         "-a <filename> : filename of matrix a (required)\n"
         "-b <filename> : filename of matrix b (required)\n"
         "-o <filename> : output-file (required)\n"
         "-V <number> : Version to be used. Default: 0\n"
         "-B<number> : Measure time for <Number> of executions. Default: 1\n"
         "-t <size>,<nonZero> : Multiply two random quadratic matrices with "
         "given size and maximum amount of <nonZero> values unequal to zero "
         "per column (-a, -b and -o not required)\n"
         "-h/--help : Help Message\n");
}

int main(int argc, char *argv[]) {
  static struct option long_options[] = {{"help", no_argument, 0, 'h'}};
  int option;
  int optionIndex = 0;
  int implementation = 0;
  int time = 0;
  int numberOfCalls = 1;
  char *a = NULL;
  char *b = NULL;
  char *o = NULL;
  int test = 0;
  uint64_t size;
  uint64_t nonZero;
  while ((option = getopt_long(argc, argv, "hV:B::a:b:o:t:", long_options,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'V':
      if (implementation == 0) {
        implementation = atoi(optarg);
        if (implementation < 0 || implementation > 2) {
          fprintf(stderr, "Invalid version number. Only 0-2 allowed!");
          return EXIT_FAILURE;
        }
        printf("Program version: %d\n", implementation);
      } else {
        fprintf(stderr, "Only one version!\n");
        return EXIT_FAILURE;
      }
      break;
    case 'B':
      if (time == 0) {
        time = 1;
        if (optarg) {
          numberOfCalls = atoi(optarg);
          if (numberOfCalls == 0) {
            numberOfCalls = 1;
            printf("Measuring time for 1 execution.\n");
          } else {
            printf("Measuring time for %d executions.\n", numberOfCalls);
          }
        } else {
          printf("Measuring time for 1 execution.\n");
        }
      } else {
        fprintf(stderr, "Only one B option allowed!\n");
        return EXIT_FAILURE;
      }
      break;
    case 'a':
      if (a == NULL) {
        a = optarg;
        printf("Matrix a: %s\n", a);
      } else {
        fprintf(stderr, "Only one a option allowed!\n");
        return EXIT_FAILURE;
      }
      break;
    case 'b':
      if (b == NULL) {
        b = optarg;
        printf("Matrix b: %s\n", b);
      } else {
        fprintf(stderr, "Only one b option allowed!\n");
        return EXIT_FAILURE;
      }
      break;
    case 'o':
      if (o == NULL) {
        o = optarg;
        printf("Output file: %s\n", o);
      } else {
        fprintf(stderr, "Only one o option allowed!\n");
        return EXIT_FAILURE;
      }
      break;
    case 'h':
      printHelpMessage();
      return EXIT_SUCCESS;
    case 't':
      if (test == 0) {
        test = 1;
        char *firstArg = strtok(optarg, ",");
        if (firstArg == NULL) {
          fprintf(stderr, "Size and nonZero required for -t\n");
          return EXIT_FAILURE;
        } else {
          size = atoll(firstArg);
        }
        char *secondArg = strtok(NULL, ",");
        if (secondArg == NULL) {
          fprintf(stderr, "Size and nonZero required for -t\n");
          return EXIT_FAILURE;
        } else {
          nonZero = atoll(secondArg);
          if (nonZero > size - 1) {
            fprintf(stderr, "NonZero has to be between 0 and size - 1!\n");
            return EXIT_FAILURE;
          }
        }
        printf("Multiplication of two random matrices with size %lu and "
               "nonZero %lu\n",
               size, nonZero);
      } else {
        fprintf(stderr, "Only one t option allowed!\n");
        return EXIT_FAILURE;
      }
      break;
    default:
      printHelpMessage();
      return EXIT_FAILURE;
    }
  }

  ELLMatrix *ellMatrixa;
  ELLMatrix *ellMatrixb;

  if (test == 0) {
    if (a == NULL || b == NULL || o == NULL) {
      fprintf(stderr, "Options a, b and o required!\n");
      return EXIT_FAILURE;
    }

    ellMatrixa = textFileToEll(a);
    ellMatrixb = textFileToEll(b);

  } else {
    ellMatrixa = generateRandomELL(nonZero, size, size, 0.5F);
    ellMatrixb = generateRandomELL(nonZero, size, size, 0.5F);
  }

  if (ellMatrixa == NULL || ellMatrixb == NULL) {
    if (ellMatrixa != NULL) {
      freeELL(ellMatrixa);
    } else {
      freeELL(ellMatrixb);
    }
    return EXIT_FAILURE;
  }

  // Check if multiplication is possible
  if (ellMatrixa->cols != ellMatrixb->cols) {
    fprintf(
        stderr,
        "Cannot multiply a matrix with %lu columns by a matrix with %lu rows\n",
        ellMatrixa->rows, ellMatrixb->rows);
    freeELL(ellMatrixa);
    freeELL(ellMatrixb);
    return EXIT_FAILURE;
  }

  ELLMatrix *output = (ELLMatrix *)calloc(1, sizeof(ELLMatrix));
  if (output == NULL) {
    fprintf(stderr, "Memory allocation failed!\n");
    freeELL(ellMatrixa);
    freeELL(ellMatrixb);
    return EXIT_FAILURE;
  }

  void (*f)(const void *, const void *, void *);

  switch (implementation) {
  case 0:
    f = &matr_mult_ellpack;
    break;
  case 1:
    f = &matr_mult_ellpack_V1;
    break;
  case 2:
    f = &matr_mult_ellpack_V2;
    break;
  case 3:
    f = &matr_mult_ellpack_V3;
    break;
  case 4:
    f = &matr_mult_ellpack_V4;
    break;
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < numberOfCalls; ++i) {
    f(ellMatrixa, ellMatrixb, output);
    if (output->cols == 0) {
      freeELL(ellMatrixa);
      freeELL(ellMatrixb);
      free(output);
      return EXIT_FAILURE;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  double sum =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  if (time == 1) {
    printf("Execution of %d function call(s) took %f seconds.\nAverage time: "
           "%f seconds\n",
           numberOfCalls, sum, sum / numberOfCalls);
  }
  if (test == 0) {
    EllToTextFile(output, o);
  }
  freeELL(ellMatrixa);
  freeELL(ellMatrixb);
  freeELL(output);
  return EXIT_SUCCESS;
}
