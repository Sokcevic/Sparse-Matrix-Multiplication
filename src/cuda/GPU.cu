#include <cstdio>
#include <cuda_runtime.h>
#include <sys/types.h>



int main() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           cudaGetErrorString(error_id));
    return -1;
  }

  if (deviceCount == 0) {
    fprintf(stderr, "There are no available CUDA devices.\n");
    return 1;
  } else {
    printf("Detected %d CUDA capable device(s).\n", deviceCount);
  }

  return 0;
}
