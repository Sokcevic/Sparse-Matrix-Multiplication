#ifndef MAIN_H
#define MAIN_H

#include "ELL.h"
#include "matrix.h"
#include <stdint.h>
#include <stdlib.h>
#include <tmmintrin.h>

void matr_mult_ellpack_v6(const void *a, const void *b, void *result);
void matr_mult_ellpack_v5(const void *a, const void *b, void *result);
void matr_mult_ellpack_v4(const void *a, const void *b, void *result);
void matr_mult_ellpack_v3(const void *a, const void *b, void *result);
void matr_mult_ellpack_v2(const void *a, const void *b, void *result);






#endif  
