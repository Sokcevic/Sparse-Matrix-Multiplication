#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct{
  uint64_t rows;
  uint64_t cols;
  float** data;
}Matrix;

void freeMatrixStruct(Matrix* a){
  for (uint64_t i = 0; i < a -> rows; i++) {
    free(a -> data[i]);
  }
  free(a -> data);
  free(a);
}

Matrix* generateRandomMatrix(float zeroRate, uint64_t rows, uint64_t cols) {
    Matrix* result = malloc(sizeof(Matrix));
    if (result == NULL) {
        return NULL;
    }
    result->rows = rows;
    result->cols = cols;
    result->data = malloc(rows * sizeof(float*));
    if (result->data == NULL) {
        free(result);
        return NULL;
    }
    for (uint64_t i = 0; i < rows; i++) {
        result->data[i] = malloc(cols * sizeof(float));
        if (result->data[i] == NULL) {
            freeMatrixStruct(result);
            return NULL;
        }
    }
    for (uint64_t i = 0; i < rows; i++) {
        for (uint64_t j = 0; j< cols; j++) {
            if ((float)rand()/(float)RAND_MAX < zeroRate) {
                result->data[i][j] = 0.0;
            } else {
                result->data[i][j] = (1000.0F * ((float)rand()/(float)RAND_MAX)) + ((float)rand()/(float)RAND_MAX);
            }
        }
    }
    return result;
}

void printMatrix(Matrix* m) {
    printf("\n");
    for (uint64_t i = 0; i < m->rows; i++) {
        for (uint64_t j = 0; j < m->cols; j++) {
            printf("%f\t", m->data[i][j]);
        }
        printf("\n");
    }
}

void matmul(Matrix* a, Matrix* b, Matrix* result){
  for (uint64_t i = 0; i < a -> rows; i++) {
    for (uint64_t j = 0; j < b -> cols; j++) {
      for (uint64_t k = 0; k < a -> cols; k++) {
        result -> data[i][j] += a -> data[i][k] * b -> data[k][j];
      }
    }
  }
}

void setup() {
    srand(42);
}

int main(void) {
    setup();
    Matrix* matrix = generateRandomMatrix(0.8F, 4, 5);
    printMatrix(matrix);
    freeMatrixStruct(matrix);
  return EXIT_SUCCESS;
}
