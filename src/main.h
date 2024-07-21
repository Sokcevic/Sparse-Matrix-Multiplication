#ifndef MAIN_H
#define MAIN_H

#include "ELL.h"
#include "matrix.h"
#include <stdint.h>
#include <stdlib.h>
#include <tmmintrin.h>

void matr_mult_ellpack(const void *a, const void *b, void *result);
void matr_mult_ellpack_V1(const void *a, const void *b, void *result);
void matr_mult_ellpack_V2(const void *a, const void *b, void *result);

#endif
