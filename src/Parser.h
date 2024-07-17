#include "ELL.h"
#ifndef SPMM_PARSER_H
#define SPMM_PARSER_H

ELLMatrix* textFileToEll(char* filename);
void EllToTextFile(ELLMatrix* ell, char* filename);

#endif //SPMM_PARSER_H
