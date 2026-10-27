#include "ELL.h"
#include "Parser.h"
#include "multiply.h"
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*MultFunction)(const void *, const void *, void *);

static const MultFunction implementations[] = {
    matr_mult_ellpack,
    matr_mult_ellpack_V1,
    matr_mult_ellpack_V2,
#ifdef WITH_CUDA
    matr_mult_ellpack_V3,
    matr_mult_ellpack_V4,
#endif
};
#ifdef WITH_CUDA
#define CUDA_VERSIONS_HELP                                                     \
  "    3: ELLPACK multiplication on the GPU (CUDA)\n"                          \
  "    4: dense multiplication on the GPU (CUDA)\n"
#else
#define CUDA_VERSIONS_HELP ""
#endif

#define NUM_IMPLEMENTATIONS                                                    \
  (sizeof(implementations) / sizeof(implementations[0]))

static void printHelpMessage(const char *programName) {
  printf("Help Message\n"
         "Usage: %s [-V <number>] [-B[<number>]] -a <filename> -b <filename> "
         "-o <filename>\n"
         "       %s [-V <number>] [-B[<number>]] -t <size>,<nonZero> "
         "[-o <filename>]\n"
         "\n"
         "-a <filename> : filename of matrix a\n"
         "-b <filename> : filename of matrix b\n"
         "-o <filename> : output file for the result a * b\n"
         "-V <number> : Version to be used. Default: 0\n"
         "    0: ELLPACK multiplication, parallel (OpenMP)\n"
         "    1: ELLPACK multiplication, sequential\n"
         "    2: dense multiplication with SIMD\n" CUDA_VERSIONS_HELP
         "-B<number> : Measure time for <number> of executions. Default: 1\n"
         "-t <size>,<nonZero> : Multiply two random square matrices with the "
         "given size and at most <nonZero> values unequal to zero per column "
         "(-a and -b not allowed, -o optional)\n"
         "-h/--help : Help Message\n"
         "\n"
         "Matrix file format:\n"
         "    <noRows>,<noCols>,<noNonZero>\n"
         "    <values>   (comma separated, '*' = don't care)\n"
         "    <indices>  (comma separated, '*' = don't care)\n"
         "\n"
         "Examples:\n"
         "    %s -a a.txt -b b.txt -o result.txt\n"
         "    %s -V 1 -B5 -a a.txt -b b.txt -o result.txt\n"
         "    %s -B10 -t 1000,50\n",
         programName, programName, programName, programName, programName);
}

static void printUsageHint(const char *programName) {
  fprintf(stderr, "Use %s --help for more information.\n", programName);
}

// Parses a non-negative decimal integer that spans the whole string
static bool parseUInt64(const char *s, uint64_t *out) {
  if (s == NULL || *s < '0' || *s > '9') { // rejects "", "-1" and "+1"
    return false;
  }
  char *end;
  errno = 0;
  unsigned long long value = strtoull(s, &end, 10);
  if (errno != 0 || *end != '\0' || value > UINT64_MAX) {
    return false;
  }
  *out = value;
  return true;
}

static bool parseTestArguments(const char *arg, uint64_t *size,
                               uint64_t *nonZero) {
  const char *comma = strchr(arg, ',');
  if (comma == NULL) {
    return false;
  }
  char sizeStr[32];
  size_t length = (size_t)(comma - arg);
  if (length >= sizeof(sizeStr)) {
    return false;
  }
  memcpy(sizeStr, arg, length);
  sizeStr[length] = '\0';
  return parseUInt64(sizeStr, size) && parseUInt64(comma + 1, nonZero);
}

int main(int argc, char *argv[]) {
  static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};
  const char *programName = argv[0];
  int option;
  uint64_t implementation = 0;
  bool versionSet = false;
  bool measureTime = false;
  uint64_t numberOfCalls = 1;
  char *a = NULL;
  char *b = NULL;
  char *o = NULL;
  bool test = false;
  uint64_t size = 0;
  uint64_t nonZero = 0;
  while ((option = getopt_long(argc, argv, "hV:B::a:b:o:t:", long_options,
                               NULL)) != -1) {
    switch (option) {
    case 'V':
      if (versionSet) {
        fprintf(stderr, "Only one V option allowed!\n");
        printUsageHint(programName);
        return EXIT_FAILURE;
      }
      versionSet = true;
      if (!parseUInt64(optarg, &implementation) ||
          implementation >= NUM_IMPLEMENTATIONS) {
        fprintf(stderr, "Invalid version \"%s\". Only 0-%zu allowed!\n",
                optarg, NUM_IMPLEMENTATIONS - 1);
        printUsageHint(programName);
        return EXIT_FAILURE;
      }
      break;
    case 'B':
      if (measureTime) {
        fprintf(stderr, "Only one B option allowed!\n");
        printUsageHint(programName);
        return EXIT_FAILURE;
      }
      measureTime = true;
      if (optarg && (!parseUInt64(optarg, &numberOfCalls) ||
                     numberOfCalls == 0)) {
        fprintf(stderr,
                "Invalid number of executions \"%s\". Has to be a positive "
                "integer (written directly after -B, e.g. -B5)!\n",
                optarg);
        printUsageHint(programName);
        return EXIT_FAILURE;
      }
      break;
    case 'a':
    case 'b':
    case 'o': {
      char **target = option == 'a' ? &a : option == 'b' ? &b : &o;
      if (*target != NULL) {
        fprintf(stderr, "Only one %c option allowed!\n", option);
        printUsageHint(programName);
        return EXIT_FAILURE;
      }
      *target = optarg;
      break;
    }
    case 'h':
      printHelpMessage(programName);
      return EXIT_SUCCESS;
    case 't':
      if (test) {
        fprintf(stderr, "Only one t option allowed!\n");
        printUsageHint(programName);
        return EXIT_FAILURE;
      }
      test = true;
      if (!parseTestArguments(optarg, &size, &nonZero) || size == 0 ||
          nonZero > size) {
        fprintf(stderr, "Invalid argument \"%s\" for -t. Expected "
                        "<size>,<nonZero> with size > 0 and "
                        "0 <= nonZero <= size!\n",
                optarg);
        printUsageHint(programName);
        return EXIT_FAILURE;
      }
      break;
    default: // getopt_long already printed an error message
      printUsageHint(programName);
      return EXIT_FAILURE;
    }
  }

  if (optind < argc) {
    fprintf(stderr, "Unexpected argument \"%s\"!\n", argv[optind]);
    printUsageHint(programName);
    return EXIT_FAILURE;
  }

  ELLMatrix *ellMatrixa;
  ELLMatrix *ellMatrixb;

  if (test) {
    if (a != NULL || b != NULL) {
      fprintf(stderr, "Options a and b cannot be combined with t!\n");
      printUsageHint(programName);
      return EXIT_FAILURE;
    }
    printf("Multiplication of two random matrices with size %" PRIu64
           " and nonZero %" PRIu64 "\n",
           size, nonZero);
    ellMatrixa = generateRandomELL(nonZero, size, size, 0.5F);
    ellMatrixb = generateRandomELL(nonZero, size, size, 0.5F);
  } else {
    if (a == NULL || b == NULL || o == NULL) {
      fprintf(stderr, "Options a, b and o required!\n");
      printUsageHint(programName);
      return EXIT_FAILURE;
    }
    ellMatrixa = textFileToEll(a);
    ellMatrixb = ellMatrixa ? textFileToEll(b) : NULL;
  }

  if (ellMatrixa == NULL || ellMatrixb == NULL) {
    freeELL(ellMatrixa);
    freeELL(ellMatrixb);
    return EXIT_FAILURE;
  }

  // Check if multiplication is possible
  if (ellMatrixa->cols != ellMatrixb->rows) {
    fprintf(stderr,
            "Cannot multiply a matrix with %" PRIu64
            " columns by a matrix with %" PRIu64 " rows\n",
            ellMatrixa->cols, ellMatrixb->rows);
    freeELL(ellMatrixa);
    freeELL(ellMatrixb);
    return EXIT_FAILURE;
  }

  MultFunction f = implementations[implementation];
  ELLMatrix output = {0};

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (uint64_t i = 0; i < numberOfCalls; ++i) {
    // discard the result of the previous call
    free(output.values);
    free(output.indicies);
    output = (ELLMatrix){0};

    f(ellMatrixa, ellMatrixb, &output);
    if (output.cols == 0) { // result untouched: multiplication failed
      fprintf(stderr, "Multiplication failed!\n");
      freeELL(ellMatrixa);
      freeELL(ellMatrixb);
      return EXIT_FAILURE;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  double sum =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  if (measureTime) {
    printf("Execution of %" PRIu64 " function call(s) of version %" PRIu64
           " took %f seconds.\nAverage time: %f seconds\n",
           numberOfCalls, implementation, sum, sum / numberOfCalls);
  }

  bool ok = o == NULL || EllToTextFile(&output, o);

  freeELL(ellMatrixa);
  freeELL(ellMatrixb);
  free(output.values);
  free(output.indicies);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
