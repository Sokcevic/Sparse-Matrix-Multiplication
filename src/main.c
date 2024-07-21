#include "ELL.h"
#include "matrix.h"
#include <stdint.h>
#include <stdlib.h>
#include <getopt.h>
#include "Parser.h"
#include <time.h>
#include <emmintrin.h>

//Ellpack parallel
void matr_mult_ellpack(const void *a, const void *b, void *result) {

    // Cast to structure
    ELLMatrix *matrix_a = transpose((ELLMatrix *)a); // transpose to read through columns
    ELLMatrix *matrix_b = (ELLMatrix *)b;

    if (matrix_a == NULL) {
        return;
    }
    //result is empty if a or b is empty
    if (matrix_a->nonZero == 0 || matrix_b->nonZero == 0) {
        ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, 0);
        freeELL(matrix_a);
        if (res) {
            *((ELLMatrix*) result) = *res;
            free(res);
        }
        return;
    }

    //calculate nonZero of result
    uint64_t maxNonZero = 0;
#pragma omp parallel for
    for (uint64_t i = 0; i < matrix_b->cols; i++) { // calculate nonZero of each column of result
        uint64_t nonZero = 0;
#pragma omp parallel for
        for (uint64_t j = 0; j < matrix_a->cols; j++) { //check if each entry of the ith column of result is 0 or increment nonZero
            uint64_t a_batch_index = j * matrix_a->nonZero;
            uint64_t b_batch_index = i * matrix_b->nonZero;
            float sum = 0;
            while (a_batch_index < (j + 1) * matrix_a->nonZero &&
                   b_batch_index < (i + 1) * matrix_b->nonZero) { // look for matches in indices of the jth column of matrix_a and the ith column of matrix_b
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

    //calculate the result
#pragma omp parallel for
    for (uint64_t i = 0; i < matrix_b->cols; i++) { // iterate through columns of result
        uint64_t result_index = i * res->nonZero;
        for (uint64_t j = 0; j < matrix_a->cols; j++) { // iterate through values in ith column
            uint64_t a_batch_index = j * matrix_a->nonZero;
            uint64_t b_batch_index = i * matrix_b->nonZero;
            float sum = 0;
            while (a_batch_index < (j + 1) * matrix_a->nonZero &&
                   b_batch_index < (i + 1) * matrix_b->nonZero) { // look for matching indices in relevant columns of matrix_a and matrix_b
                if (matrix_a->values[a_batch_index] == 0 ||
                    matrix_b->values[b_batch_index] == 0) {
                    break;
                }
                uint64_t a_next_index = matrix_a->indicies[a_batch_index];
                uint64_t b_next_index = matrix_b->indicies[b_batch_index];
                if (a_next_index == b_next_index) {
                    sum += matrix_a->values[a_batch_index] * matrix_b->values[b_batch_index];
                    a_batch_index++;
                    b_batch_index++;
                } else if (a_next_index < b_next_index) {
                    a_batch_index++;
                } else {
                    b_batch_index++;
                }
            }
            if (sum != 0) { // write value
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
    ELLMatrix *matrix_a = transpose((ELLMatrix *)a); // transpose to read through columns
    ELLMatrix *matrix_b = (ELLMatrix *)b;

    if (matrix_a == NULL) {
        return;
    }
    //result is empty if a or b is empty
    if (matrix_a->nonZero == 0 || matrix_b->nonZero == 0) {
        ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, 0);
        freeELL(matrix_a);
        if (res) {
            *((ELLMatrix*) result) = *res;
            free(res);
        }
        return;
    }

    //calculate nonZero of result
    uint64_t maxNonZero = 0;
    for (uint64_t i = 0; i < matrix_b->cols; i++) { // calculate nonZero of each column of result
        uint64_t nonZero = 0;
        for (uint64_t j = 0; j < matrix_a->cols; j++) { //check if each entry of the ith column of result is 0 or increment nonZero
            uint64_t a_batch_index = j * matrix_a->nonZero;
            uint64_t b_batch_index = i * matrix_b->nonZero;
            float sum = 0;
            while (a_batch_index < (j + 1) * matrix_a->nonZero &&
                   b_batch_index < (i + 1) * matrix_b->nonZero) { // look for matches in indices of the jth column of matrix_a and the ith column of matrix_b
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

    //calculate the result
    for (uint64_t i = 0; i < matrix_b->cols; i++) { // iterate through columns of result
        uint64_t result_index = i * res->nonZero;
        for (uint64_t j = 0; j < matrix_a->cols; j++) { // iterate through values in ith column
            uint64_t a_batch_index = j * matrix_a->nonZero;
            uint64_t b_batch_index = i * matrix_b->nonZero;
            float sum = 0;
            while (a_batch_index < (j + 1) * matrix_a->nonZero &&
                   b_batch_index < (i + 1) * matrix_b->nonZero) { // look for matching indices in relevant columns of matrix_a and matrix_b
                if (matrix_a->values[a_batch_index] == 0 ||
                    matrix_b->values[b_batch_index] == 0) {
                    break;
                }
                uint64_t a_next_index = matrix_a->indicies[a_batch_index];
                uint64_t b_next_index = matrix_b->indicies[b_batch_index];
                if (a_next_index == b_next_index) {
                    sum += matrix_a->values[a_batch_index] * matrix_b->values[b_batch_index];
                    a_batch_index++;
                    b_batch_index++;
                } else if (a_next_index < b_next_index) {
                    a_batch_index++;
                } else {
                    b_batch_index++;
                }
            }
            if (sum != 0) { // write value
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
  ELLMatrix *matrix_b = transpose((ELLMatrix *)b); // transpose to read through lines

    if (matrix_b == NULL) {
        return;
    }

    // result is empty if matrix_a or matrix_b is empty
    if (matrix_a->nonZero == 0 || matrix_b->nonZero == 0) {
        ELLMatrix *res = createELL(matrix_a->cols, matrix_b->cols, 0);
        freeELL(matrix_b);
        if (res) {
            *((ELLMatrix*) result) = *res;
            free(res);
        }
        return;
    }

  // cast to normal matrices
  Matrix *ma = fromELL(matrix_a);
  Matrix *mb = fromELL(matrix_b);
  // initialize result
  Matrix *mr = createMatrix(ma->rows, mb->rows);

  // iterate through entries of result and calculate them
  for (uint64_t i = 0; i < matrix_a->rows; i++) {
    for (uint64_t j = 0; j < matrix_b->rows; j++) {
      if (matrix_a->cols > 3) {
        __m128 sum = _mm_setzero_ps(); // Initialize sum to zero
        for (uint64_t k = 0; k < matrix_a->cols - (matrix_a->cols % 4);
             k += 4) {
          __m128 a_values = _mm_loadu_ps(&ma->data[i][k]);
          __m128 b_values = _mm_loadu_ps(&mb->data[j][k]);
          __m128 product = _mm_mul_ps(a_values, b_values);
          sum = _mm_add_ps(sum, product);
        }
        float temp[4];
        _mm_storeu_ps(temp, sum);

        mr->data[i][j] += temp[0] + temp[1] + temp[2] + temp[3];
      }
      // calculate remaining values if matrix_a->cols is not dividable by 4
      for (int k = matrix_a->cols - (matrix_a->cols % 4); k < matrix_a->cols;
           k++) {
        mr->data[i][j] += ma->data[i][k] * mb->data[j][k];
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

void printHelpMessage() {
    printf("Help Message\n"
           "-a <filename> : filename of matrix a (required)\n"
           "-b <filename> : filename of matrix b (required)\n"
           "-o <filename> : output-file (required)\n"
           "-V <number> : Version to be used. Default: 0\n"
           "-B<number> : Measure time for <Number> of executions. Default: 1\n"
           "-t <size>,<nonZero> : Multiply two random quadratic matrices with given size and maximum amount of <nonZero> values unequal to zero per column (-a, -b and -o not required)\n"
           "-h/--help : Help Message\n"
           );
}

int main(int argc, char *argv[]) {
    static struct option long_options[] = {
            {"help", no_argument, 0,'h'}
    };
    int option;
    int optionIndex = 0;
    int implementation = 0;
    int time = 0;
    int numberOfCalls = 1;
    char* a = NULL;
    char* b = NULL;
    char* o = NULL;
    int test = 0;
    uint64_t size;
    uint64_t nonZero;
    while ((option = getopt_long(argc, argv, "hV:B::a:b:o:t:", long_options, &optionIndex)) != -1) {
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
                    char* firstArg = strtok(optarg, ",");
                    if (firstArg == NULL) {
                        fprintf(stderr, "Size and nonZero required for -t\n");
                        return EXIT_FAILURE;
                    } else {
                        size = atoll(firstArg);
                    }
                    char* secondArg = strtok(NULL, ",");
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
                    printf("Multiplication of two random matrices with size %lu and nonZero %lu\n", size, nonZero);
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

    ELLMatrix* ellMatrixa;
    ELLMatrix* ellMatrixb;

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
        } if (ellMatrixb != NULL){
            freeELL(ellMatrixb);
        }
        return EXIT_FAILURE;
    }

    // Check if multiplication is possible
    if (ellMatrixa->cols != ellMatrixb->cols) {
        fprintf(stderr, "Cannot multiply a matrix with %lu columns by a matrix with %lu rows\n", ellMatrixa->rows, ellMatrixb->rows);
        freeELL(ellMatrixa);
        freeELL(ellMatrixb);
        return EXIT_FAILURE;
    }

    ELLMatrix* output = calloc(1, sizeof(ELLMatrix));
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
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < numberOfCalls; ++i) {
        f(ellMatrixa, ellMatrixb, output);
        if (output->cols == 0) { //check if multiplication did not work
            freeELL(ellMatrixa);
            freeELL(ellMatrixb);
            free(output);
            return EXIT_FAILURE;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double sum = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (time == 1) {
        printf("Execution of %d function call(s) took %f seconds.\nAverage time: %f seconds\n", numberOfCalls, sum, sum/numberOfCalls);
    }
    if (test == 0) {
        EllToTextFile(output, o);
    }
    freeELL(ellMatrixa);
    freeELL(ellMatrixb);
    freeELL(output);
    return EXIT_SUCCESS;
}
