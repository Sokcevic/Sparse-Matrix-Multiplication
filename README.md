# Sparse Matrix Multiplication (ELLPACK, CPU + CUDA)

Sparse matrix × sparse matrix multiplication in C and CUDA, with matrices stored in the **ELLPACK** format. The project includes several CPU and GPU implementations so they can be benchmarked against each other and against dense multiplication.

## ELLPACK format

Dense multiplication of sparse matrices wastes almost all of its work multiplying zeros. ELLPACK stores only the non-zero values plus their indices. This repo uses a column-major variant: every column has `nonZero` slots (the maximum number of non-zeros in any column), and each slot stores a value and its row index. The non-zeros of a column come first, sorted by row; the remaining slots are zero padding ([`src/ELL.h`](src/ELL.h)).

To compute `C = A × B`, `A` is first transposed in ELLPACK form, so each of its rows becomes a sorted column. Every entry `C[i][j]` is then a **sorted merge (sparse dot product)** of column `i` of `Aᵀ` and column `j` of `B`. The cost depends only on the non-zeros, not on the matrix dimensions.

## Implementations

Select an implementation with `-V <n>`:

| `-V` | Implementation | Where |
|---|---|---|
| 0 | ELLPACK × ELLPACK, parallelized over result columns with **OpenMP** (default) | [`src/multiply.c`](src/multiply.c) |
| 1 | ELLPACK × ELLPACK, sequential (reference) | [`src/multiply.c`](src/multiply.c) |
| 2 | Converted to dense, then dense multiplication vectorized with **SSE SIMD** intrinsics | [`src/multiply.c`](src/multiply.c) |
| 3 | ELLPACK × ELLPACK on the **GPU (CUDA)** | [`src/cuda/multiply.cu`](src/cuda/multiply.cu) |
| 4 | Dense multiplication on the **GPU (CUDA)** | [`src/cuda/multiply.cu`](src/cuda/multiply.cu) |

**CUDA ELLPACK kernel (V3).** The output size isn't known in advance, so it runs in two passes:
1. `countNonZeroKernel` launches one thread per result column. Each thread counts that column's non-zeros, and the counts determine the result's `nonZero` width.
2. After the result buffer is allocated, `ellMulKernel` launches one thread per column again. Each thread computes the column's sparse dot products and writes the non-zero results directly in ELLPACK layout.

Device allocation, transfers and every CUDA call are error-checked, and partially allocated device memory is cleaned up on failure.

The CPU build is restricted to SSE (no AVX, MMX or x87), so the SIMD path is portable to any x86-64 machine.

**Results (summary):** on sparse inputs (1–10 % density), ELLPACK multiplication is much faster than dense multiplication. Around 50 % density, the dense SIMD version wins. The GPU's advantage grows with matrix size.

## Build

```sh
cd src
make            # CPU build (gcc or clang with OpenMP) -> ./main
make cuda       # additionally builds the GPU versions (needs nvcc) -> ./main_cuda
```

## Usage

```sh
./main -a a.txt -b b.txt -o result.txt          # multiply two matrices from files
./main -V 1 -B5 -a a.txt -b b.txt -o result.txt # version 1, average over 5 runs
./main -B10 -t 1000,50                          # benchmark two random 1000x1000 matrices,
                                                # at most 50 non-zeros per column
./main_cuda -V 3 -B10 -t 4000,40                # GPU ELLPACK version
```

Matrix file format (ELLPACK, column-major, `*` = padding):

```
<rows>,<cols>,<nonZero>
<values, comma separated>
<row indices, comma separated>
```

## Tests

```sh
make test        # CPU versions: parser edge cases + all versions checked against each other
make test-cuda   # same, including the GPU versions
```

## Authors

Lukas Šokčević (dense/SIMD multiplication, format conversions, CUDA implementations), Paul Fuchs (ELLPACK multiplication and transpose), Luis Schreiber (file I/O, parser, CLI).
