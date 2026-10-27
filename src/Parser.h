#ifndef SPMM_PARSER_H
#define SPMM_PARSER_H

#include "ELL.h"
#include <stdbool.h>

ELLMatrix *textFileToEll(const char *filename);
bool EllToTextFile(const ELLMatrix *ell, const char *filename);

#endif // SPMM_PARSER_H
