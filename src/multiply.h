#ifndef MULTIPLY_H
#define MULTIPLY_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * All implementations take two ELLMatrix pointers a and b with
 * a->cols == b->rows and write the ELLMatrix a * b into result. The caller
 * owns the arrays of result afterwards. On failure (out of memory) result is
 * left untouched.
 */

// ELLPACK multiplication, parallelized with OpenMP
void matr_mult_ellpack(const void *a, const void *b, void *result);
// ELLPACK multiplication, sequential
void matr_mult_ellpack_V1(const void *a, const void *b, void *result);
// dense multiplication with SIMD (SSE)
void matr_mult_ellpack_V2(const void *a, const void *b, void *result);

#ifdef WITH_CUDA
// GPU implementations in cuda/multiply.cu (built with make cuda)

// ELLPACK multiplication on the GPU
void matr_mult_ellpack_V3(const void *a, const void *b, void *result);
// dense multiplication on the GPU
void matr_mult_ellpack_V4(const void *a, const void *b, void *result);
#endif

#ifdef __cplusplus
}
#endif

#endif // MULTIPLY_H
