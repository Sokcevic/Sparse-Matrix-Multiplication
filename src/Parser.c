#include "Parser.h"
#include "ELL.h"
#include <stdio.h>

ELLMatrix *textFileToEll(char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Could not open file %s\n", filename);
    return NULL;
  }
  if (feof(file)) {
    fclose(file);
    fprintf(stderr, "File %s is empty\n", filename);
    return NULL;
  }
  uint64_t rows;
  uint64_t cols;
  uint64_t nonZero;
  if (fscanf(file, "%lu,%lu,%lu\n", &rows, &cols, &nonZero) != 3) {
    fclose(file);
    fprintf(stderr, "Format of first line was wrong\n");
    return NULL;
  }
  if (rows == 0 || cols == 0) {
    fclose(file);
    fprintf(stderr,
            "Invalid arguments. No matrix with 0 rows or 0 columns allowed\n");
    return NULL;
  }
  uint64_t size = cols * nonZero;
  ELLMatrix *result = createELL(rows, cols, nonZero);
  if (result == NULL) {
    fclose(file);
    return NULL;
  }
  if (size != 0) {
    for (uint64_t i = 0; i < size - 1; i++) {
      char peek = fgetc(file);
      if (peek == EOF || peek == '\n') {
        fclose(file);
        freeELL(result);
        fprintf(stderr, "Values array too short!\n");
        return NULL;
      }
      if (peek == '*') {
        fgetc(file);
      } else {
        ungetc(peek, file);
        if (1 != fscanf(file, "%f,", &(result->values[i]))) {
          fclose(file);
          freeELL(result);
          fprintf(stderr, "Error while reading Values!\n");
          return NULL;
        }
      }
    }
  }
  // read last entry of array (no comma after)
  char lastChar = fgetc(file);
  if (lastChar == EOF) {
    fclose(file);
    freeELL(result);
    fprintf(stderr, "Values array too short!\n");
    return NULL;
  }
  if (lastChar != '*') {
    ungetc(lastChar, file);
    if (1 != fscanf(file, "%f", &(result->values[size - 1]))) {
      fclose(file);
      freeELL(result);
      fprintf(stderr, "Error while reading file!\n");
      return NULL;
    }
  }

  fgetc(file);

  uint64_t lastIndex;
  for (uint64_t i = 0; i < size - 1; i++) {
    if (i % nonZero == 0)
      lastIndex = 0;
    char peek = fgetc(file);
    if (peek == EOF) {
      fclose(file);
      freeELL(result);
      fprintf(stderr, "Indices array too short!\n");
      return NULL;
    }
    if (peek == '*') {
      fgetc(file);
    } else {
      ungetc(peek, file);
      if (1 != fscanf(file, "%lu,", &(result->indicies[i]))) {
        fclose(file);
        freeELL(result);
        fprintf(stderr, "Error while reading file!\n");
        return NULL;
      }
      if (result->indicies[i] < lastIndex || result->indicies[i] >= rows) {
        fclose(file);
        freeELL(result);
        fprintf(stderr, "Wrong indices!\n");
        return NULL;
      }
      lastIndex = result->indicies[i];
    }
  }
  // read last entry of array (no comma after)
  lastChar = fgetc(file);
  if (lastChar == EOF) {
    fclose(file);
    freeELL(result);
    fprintf(stderr, "Indices array too short!\n");
    return NULL;
  }
  if (lastChar != '*') {
    ungetc(lastChar, file);
    if (1 != fscanf(file, "%lu", &(result->indicies[size - 1]))) {
      fclose(file);
      freeELL(result);
      fprintf(stderr, "Error while reading file!\n");
      return NULL;
    }
    if (result->indicies[size - 1] < lastIndex ||
        result->indicies[size - 1] >= rows) {
      fclose(file);
      freeELL(result);
      fprintf(stderr, "Wrong indices!\n");
      return NULL;
    }
  }

  fclose(file);
  return result;
}

void EllToTextFile(ELLMatrix *ell, char *filename) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    fprintf(stderr, "Could not open file %s\n", filename);
    return;
  }
  fprintf(file, "%lu,%lu,%lu", ell->rows, ell->cols, ell->nonZero);
  uint64_t size = ell->cols * ell->nonZero;
  if (size == 0) {
    fclose(file);
    return;
  }
  fprintf(file, "\n");
  for (uint64_t i = 0; i < size - 1; i++) {
    float next = ell->values[i];
    if (next == 0.0) {
      fprintf(file, "*,");
    } else {
      fprintf(file, "%f,", next);
    }
  }
  float lastF = ell->values[size - 1];
  if (lastF == 0.0) {
    fprintf(file, "*\n");
  } else {
    fprintf(file, "%f\n", lastF);
  }

  for (uint64_t i = 0; i < size - 1; i++) {
    if (ell->values[i] == 0.0) {
      fprintf(file, "*,");
    } else {
      fprintf(file, "%lu,", ell->indicies[i]);
    }
  }
  if (ell->values[size - 1] == 0.0) {
    fprintf(file, "*");
  } else {
    fprintf(file, "%lu", ell->indicies[size - 1]);
  }
  fclose(file);
}
