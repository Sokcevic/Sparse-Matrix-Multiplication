#include "ELL.h"
#include "Parser.h"
#include "matrix.h"
#include "multiply.h"
#include <stdbool.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef void (*MultFunction)(const void *, const void *, void *);

static int failures = 0;

#define CHECK(condition, ...)                                                  \
  do {                                                                         \
    if (!(condition)) {                                                        \
      failures++;                                                              \
      fprintf(stderr, "FAILED %s:%d: ", __FILE__, __LINE__);                   \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
    }                                                                          \
  } while (0)

static Matrix *generateRandomMatrix(float zeroRate, uint64_t rows,
                                    uint64_t cols) {
  Matrix *result = createMatrix(rows, cols);
  for (uint64_t i = 0; i < rows; i++) {
    for (uint64_t j = 0; j < cols; j++) {
      if ((float)rand() / (float)RAND_MAX < zeroRate) {
        result->data[i][j] = 0.0F;
      } else {
        result->data[i][j] = (1000.0F * ((float)rand() / (float)RAND_MAX)) +
                             ((float)rand() / (float)RAND_MAX);
      }
    }
  }
  return result;
}

// largest difference between a and b relative to the largest value of a
static float relativeDifference(const Matrix *a, const Matrix *b) {
  float max = 0.0F;
  for (uint64_t i = 0; i < a->rows; i++) {
    for (uint64_t j = 0; j < a->cols; j++) {
      float value = a->data[i][j] > 0 ? a->data[i][j] : -a->data[i][j];
      max = value > max ? value : max;
    }
  }
  float diff = matrixEqual(a, b);
  return max == 0 ? diff : diff / max;
}

// checks that every column is sorted and padding comes last (see ELL.h)
static bool satisfiesInvariant(const ELLMatrix *m) {
  for (uint64_t col = 0; col < m->cols; col++) {
    bool padding = false;
    for (uint64_t k = 0; k < m->nonZero; k++) {
      uint64_t i = col * m->nonZero + k;
      if (m->values[i] == 0) {
        padding = true;
      } else if (padding || m->indicies[i] >= m->rows ||
                 (k > 0 && m->indicies[i] <= m->indicies[i - 1])) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Multiplies random matrices with f and compares the result with the naive
 * dense multiplication. tolerance is the allowed relative error.
 */
static void testMultiplication(const char *name, MultFunction f,
                               float tolerance) {
  const uint64_t shapes[][3] = {{1, 1, 1},   {1, 5, 1},    {5, 1, 5},
                                {4, 4, 4},   {3, 7, 2},    {17, 33, 9},
                                {64, 64, 64}, {100, 40, 120}};
  const float zeroRates[] = {0.0F, 0.5F, 0.9F, 1.0F};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    for (size_t z = 0; z < sizeof(zeroRates) / sizeof(zeroRates[0]); z++) {
      uint64_t n = shapes[s][0], m = shapes[s][1], p = shapes[s][2];
      Matrix *a = generateRandomMatrix(zeroRates[z], n, m);
      Matrix *b = generateRandomMatrix(zeroRates[z], m, p);
      Matrix *expected = createMatrix(n, p);
      matmul(a, b, expected);

      ELLMatrix *al = toELL(a);
      ELLMatrix *bl = toELL(b);
      ELLMatrix cl = {0};
      f(al, bl, &cl);

      CHECK(cl.rows == n && cl.cols == p,
            "%s: result of %" PRIu64 "x%" PRIu64 " * %" PRIu64 "x%" PRIu64 " has wrong size %" PRIu64 "x%" PRIu64 "", name,
            n, m, m, p, cl.rows, cl.cols);
      if (cl.rows == n && cl.cols == p) {
        CHECK(satisfiesInvariant(&cl), "%s: result violates ELL invariant",
              name);
        Matrix *actual = fromELL(&cl);
        float diff = relativeDifference(expected, actual);
        CHECK(diff <= tolerance,
              "%s: %" PRIu64 "x%" PRIu64 " * %" PRIu64 "x%" PRIu64 " (zero rate %.1f) differs by %g", name, n,
              m, m, p, zeroRates[z], diff);
        freeMatrix(actual);
      }

      free(cl.values);
      free(cl.indicies);
      freeELL(al);
      freeELL(bl);
      freeMatrix(a);
      freeMatrix(b);
      freeMatrix(expected);
    }
  }
}

static void testTranspose(void) {
  for (int i = 0; i < 10; i++) {
    Matrix *a = generateRandomMatrix(0.6F, 30, 35);
    ELLMatrix *al = toELL(a);
    ELLMatrix *at = transpose(al);
    CHECK(satisfiesInvariant(at), "transpose violates ELL invariant");
    Matrix *atm = fromELL(at);
    bool ok = atm->rows == a->cols && atm->cols == a->rows;
    for (uint64_t r = 0; ok && r < a->rows; r++)
      for (uint64_t c = 0; ok && c < a->cols; c++)
        ok = a->data[r][c] == atm->data[c][r];
    CHECK(ok, "transpose is wrong");
    freeMatrix(atm);
    freeELL(at);
    freeELL(al);
    freeMatrix(a);
  }
}

static void testRandomELL(void) {
  for (int i = 0; i < 10; i++) {
    ELLMatrix *m = generateRandomELL(8, 50, 20, 0.5F);
    CHECK(satisfiesInvariant(m), "generateRandomELL violates ELL invariant");
    freeELL(m);
  }
  ELLMatrix *m = generateRandomELL(10, 10, 10, 0.0F);
  CHECK(satisfiesInvariant(m), "full generateRandomELL violates invariant");
  freeELL(m);
}

static char tempFile[] = "/tmp/ell_test_XXXXXX";

static void writeTempFile(const char *content) {
  FILE *file = fopen(tempFile, "w");
  fputs(content, file);
  fclose(file);
}

static void testParser(void) {
  // example from the assignment
  writeTempFile("4,4,2\n5,0.5,6,7,*,*,3,*\n0,2,1,2,*,*,3,*\n");
  ELLMatrix *example = textFileToEll(tempFile);
  CHECK(example != NULL, "could not read example");
  if (example) {
    Matrix *dense = fromELL(example);
    const float expected[4][4] = {
        {5, 0, 0, 0}, {0, 6, 0, 0}, {0.5F, 7, 0, 0}, {0, 0, 0, 3}};
    bool ok = true;
    for (int r = 0; r < 4; r++)
      for (int c = 0; c < 4; c++)
        ok = ok && dense->data[r][c] == expected[r][c];
    CHECK(ok, "example was read incorrectly");
    freeMatrix(dense);
    freeELL(example);
  }

  const char *valid[] = {
      "2,2,1\n1,1\n1,0\n",                   // one value per column
      "4,4,0\n",                             // zero matrix without lines
      "4,4,0\n\n\n",                         // zero matrix with empty lines
      "3,1,3\n*,2,*\n7,1,*\n",               // padding first, index ignored
      "2,2,2\r\n1,2,3,4\r\n0,1,0,1\r\n",     // windows line endings
      "2,1,2\n1,0\n0,0\n",                   // explicit 0 is padding
  };
  for (size_t i = 0; i < sizeof(valid) / sizeof(valid[0]); i++) {
    writeTempFile(valid[i]);
    ELLMatrix *m = textFileToEll(tempFile);
    CHECK(m != NULL, "valid file %zu was rejected", i);
    if (m) {
      CHECK(satisfiesInvariant(m), "file %zu violates ELL invariant", i);
    }
    freeELL(m);
  }

  const char *invalid[] = {
      "",                            // empty file
      "0,4,0\n",                     // no rows
      "-1,4,1\n1,2,3,4\n0,0,0,0\n",  // negative size
      "2,2,1\n1,1\n",                // indices missing
      "2,2,1\n1\n0,0\n",             // too few values
      "2,2,1\n1,1,1\n0,0\n",         // too many values
      "2,2,1\n1,1\n0,0,0\n",         // too many indices
      "2,2,1\n1,x\n0,0\n",           // value is no number
      "2,2,1\n1,1\n0,2\n",           // index out of range
      "2,2,1\n1,1\n0,*\n",           // value without index
      "2,1,2\n1,1\n1,0\n",           // indices not increasing
      "2,1,2\n1,1\n1,1\n",           // duplicate index
      "2,2,1\n1,1\n0,0\n5\n",        // trailing content
      "2,2,1\n1,nan\n0,0\n",         // not finite
  };
  for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
    writeTempFile(invalid[i]);
    ELLMatrix *m = textFileToEll(tempFile);
    CHECK(m == NULL, "invalid file %zu was accepted", i);
    freeELL(m);
  }

  // write and read back
  ELLMatrix *original = generateRandomELL(5, 40, 30, 0.5F);
  original->values[0] = 0.1F; // needs full precision
  CHECK(EllToTextFile(original, tempFile), "could not write file");
  ELLMatrix *read = textFileToEll(tempFile);
  CHECK(read != NULL && ellEquals(original, read) == 0.0F,
        "matrix changed after writing and reading");
  freeELL(read);
  freeELL(original);
}

int main(void) {
  srand(69);
  int fd = mkstemp(tempFile);
  if (fd == -1) {
    perror("mkstemp");
    return EXIT_FAILURE;
  }
  close(fd);

  testTranspose();
  testRandomELL();
  testParser();
  testMultiplication("matr_mult_ellpack", matr_mult_ellpack, 0.0F);
  testMultiplication("matr_mult_ellpack_V1", matr_mult_ellpack_V1, 0.0F);
  testMultiplication("matr_mult_ellpack_V2", matr_mult_ellpack_V2, 1e-5F);
#ifdef WITH_CUDA
  // the GPU may use fused multiply-add, so the results are not bit-identical
  testMultiplication("matr_mult_ellpack_V3", matr_mult_ellpack_V3, 1e-5F);
  testMultiplication("matr_mult_ellpack_V4", matr_mult_ellpack_V4, 1e-5F);
#endif

  unlink(tempFile);
  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  printf("All tests passed.\n");
  return EXIT_SUCCESS;
}
