#include "hapi.h"

#define BLOCK_SIZE 256

__global__ void vecAdd(double* C, double* A, double* B, int n) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n) {
    C[id] = A[id] + B[id];
  }
}

void cudaVecAdd(int n, double* h_A, double* h_B, double* h_C, double* d_A,
                double* d_B, double* d_C, cudaStream_t stream) {
  dim3 block_dim(BLOCK_SIZE);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x);

  vecAdd<<<grid_dim, block_dim, 0, stream>>>(d_C, d_A, d_B, n);
  hapiCheck(cudaPeekAtLastError());
}
