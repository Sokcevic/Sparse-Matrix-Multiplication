#include "Parser.h"
#include "ELL.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKEN_LENGTH 64
#define TOKEN_TOO_LONG -2

/**
 * Reads the next comma separated token into buf (without surrounding
 * whitespace). Returns the character that ended the token (',', '\n' or EOF)
 * or TOKEN_TOO_LONG.
 */
static int readToken(FILE *file, char *buf) {
  size_t length = 0;
  int c;
  while ((c = fgetc(file)) != EOF && c != ',' && c != '\n') {
    if (isspace(c)) {
      continue; // ignores blanks and '\r' of windows line endings
    }
    if (length == MAX_TOKEN_LENGTH - 1) {
      return TOKEN_TOO_LONG;
    }
    buf[length++] = (char)c;
  }
  buf[length] = '\0';
  return c;
}

static bool parseUInt(const char *token, uint64_t *out) {
  if (!isdigit((unsigned char)token[0])) { // rejects "", "-1" and "+1"
    return false;
  }
  char *end;
  errno = 0;
  unsigned long long value = strtoull(token, &end, 10);
  if (errno != 0 || *end != '\0' || value > UINT64_MAX) {
    return false;
  }
  *out = value;
  return true;
}

static bool parseFloat(const char *token, float *out) {
  if (token[0] == '\0') {
    return false;
  }
  char *end;
  errno = 0;
  float value = strtof(token, &end);
  if (errno != 0 || *end != '\0' || !isfinite(value)) {
    return false;
  }
  *out = value;
  return true;
}

static bool isDontCare(const char *token) { return strcmp(token, "*") == 0; }

/**
 * Checks the delimiter after entry i of a line with size entries: entries are
 * separated by ',' and the line has to end after the last one.
 */
static bool checkDelimiter(int delimiter, uint64_t i, uint64_t size,
                           const char *line) {
  if (delimiter == TOKEN_TOO_LONG) {
    fprintf(stderr, "Entry %" PRIu64 " of the %s is too long!\n", i + 1, line);
    return false;
  }
  if (i + 1 < size && delimiter != ',') {
    fprintf(stderr,
            "The %s line is too short: expected %" PRIu64
            " entries but found %" PRIu64 "!\n",
            line, size, i + 1);
    return false;
  }
  if (i + 1 == size && delimiter == ',') {
    fprintf(stderr, "The %s line has more than %" PRIu64 " entries!\n", line,
            size);
    return false;
  }
  return true;
}

/**
 * Moves the non-zero entries of every column to the front so the matrix
 * satisfies the invariant documented in ELL.h.
 */
static void compactColumns(ELLMatrix *m) {
  for (uint64_t col = 0; col < m->cols; col++) {
    uint64_t start = col * m->nonZero;
    uint64_t next = start;
    for (uint64_t i = start; i < start + m->nonZero; i++) {
      if (m->values[i] != 0) {
        m->values[next] = m->values[i];
        m->indicies[next] = m->indicies[i];
        next++;
      }
    }
    for (; next < start + m->nonZero; next++) {
      m->values[next] = 0;
      m->indicies[next] = 0;
    }
  }
}

static ELLMatrix *parseFile(FILE *file) {
  char token[MAX_TOKEN_LENGTH];

  // first line: <noRows>,<noCols>,<noNonZero>
  uint64_t header[3];
  for (int i = 0; i < 3; i++) {
    int delimiter = readToken(file, token);
    bool validDelimiter =
        i < 2 ? delimiter == ',' : delimiter == '\n' || delimiter == EOF;
    if (!validDelimiter || !parseUInt(token, &header[i])) {
      fprintf(stderr, "First line has to be <noRows>,<noCols>,<noNonZero> "
                      "with non-negative integers!\n");
      return NULL;
    }
  }
  uint64_t rows = header[0];
  uint64_t cols = header[1];
  uint64_t nonZero = header[2];
  if (rows == 0 || cols == 0) {
    fprintf(stderr,
            "Invalid arguments. No matrix with 0 rows or 0 columns allowed\n");
    return NULL;
  }

  ELLMatrix *result = createELL(rows, cols, nonZero);
  if (result == NULL) {
    return NULL;
  }
  uint64_t size = cols * nonZero;

  // second line: values ('*' = don't care, stored as 0)
  for (uint64_t i = 0; i < size; i++) {
    int delimiter = readToken(file, token);
    if (!checkDelimiter(delimiter, i, size, "values")) {
      freeELL(result);
      return NULL;
    }
    if (!isDontCare(token) && !parseFloat(token, &result->values[i])) {
      fprintf(stderr, "Value %" PRIu64 " (\"%s\") is not a valid float!\n",
              i + 1, token);
      freeELL(result);
      return NULL;
    }
  }

  // third line: indices, only relevant for non-zero values
  uint64_t lastIndex = 0;
  for (uint64_t i = 0; i < size; i++) {
    int delimiter = readToken(file, token);
    if (!checkDelimiter(delimiter, i, size, "indices")) {
      freeELL(result);
      return NULL;
    }
    bool firstInColumn = i % nonZero == 0;
    if (firstInColumn) {
      lastIndex = 0;
    }
    if (result->values[i] == 0) {
      continue; // padding, its index is don't care
    }
    uint64_t index;
    if (!parseUInt(token, &index)) {
      fprintf(stderr, "Index %" PRIu64 " (\"%s\") is not a valid index!\n",
              i + 1, token);
      freeELL(result);
      return NULL;
    }
    if (index >= rows) {
      fprintf(stderr,
              "Index %" PRIu64 " (%" PRIu64
              ") is out of range for a matrix with %" PRIu64 " rows!\n",
              i + 1, index, rows);
      freeELL(result);
      return NULL;
    }
    // lastIndex is the row of the previous non-zero value + 1
    if (index < lastIndex) {
      fprintf(stderr,
              "Indices of column %" PRIu64
              " have to be strictly increasing!\n",
              i / nonZero);
      freeELL(result);
      return NULL;
    }
    lastIndex = index + 1;
    result->indicies[i] = index;
  }

  // only whitespace may follow
  int c;
  while ((c = fgetc(file)) != EOF) {
    if (!isspace(c)) {
      fprintf(stderr, "Unexpected content after the indices line!\n");
      freeELL(result);
      return NULL;
    }
  }

  compactColumns(result);
  return result;
}

ELLMatrix *textFileToEll(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Could not open file %s: %s\n", filename, strerror(errno));
    return NULL;
  }
  ELLMatrix *result = parseFile(file);
  if (result == NULL) {
    fprintf(stderr, "Could not read matrix from %s\n", filename);
  } else if (ferror(file)) {
    fprintf(stderr, "Error while reading %s\n", filename);
    freeELL(result);
    result = NULL;
  }
  fclose(file);
  return result;
}

/**
 * Writes the shortest representation of value that reads back as the same
 * float.
 */
static void writeFloat(FILE *file, float value) {
  char buf[MAX_TOKEN_LENGTH];
  for (int precision = 6; precision < 9; precision++) {
    snprintf(buf, sizeof(buf), "%.*g", precision, value);
    if (strtof(buf, NULL) == value) {
      fputs(buf, file);
      return;
    }
  }
  fprintf(file, "%.9g", value);
}

bool EllToTextFile(const ELLMatrix *ell, const char *filename) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    fprintf(stderr, "Could not open file %s: %s\n", filename, strerror(errno));
    return false;
  }
  fprintf(file, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", ell->rows, ell->cols,
          ell->nonZero);
  uint64_t size = ell->cols * ell->nonZero;

  for (uint64_t i = 0; i < size; i++) {
    if (i != 0) {
      fputc(',', file);
    }
    if (ell->values[i] == 0) {
      fputc('*', file);
    } else {
      writeFloat(file, ell->values[i]);
    }
  }
  fputc('\n', file);

  for (uint64_t i = 0; i < size; i++) {
    if (i != 0) {
      fputc(',', file);
    }
    if (ell->values[i] == 0) {
      fputc('*', file);
    } else {
      fprintf(file, "%" PRIu64, ell->indicies[i]);
    }
  }
  fputc('\n', file);

  bool ok = !ferror(file);
  if (fclose(file) != 0) {
    ok = false;
  }
  if (!ok) {
    fprintf(stderr, "Error while writing %s\n", filename);
  }
  return ok;
}
