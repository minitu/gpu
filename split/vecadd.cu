#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>

#define BLOCK_SIZE 256
#define WARMUP 0
#define WARMUP_COUNT 10
#define VERIFY 0

__global__ void warmup(double* a, double* b, int n) {
  int i = BLOCK_SIZE * blockIdx.x + threadIdx.x;

  if (i < n) {
    a[i] = a[i] + a[i];
    a[i] = a[i] / 2;
    b[i] = b[i] + b[i];
    b[i] = b[i] / 2;
  }
}

__global__ void vecadd(double* a, double* b, double* c, int n) {
  int i = BLOCK_SIZE * blockIdx.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char** argv) {
  int n = 1024;
  int split = 1;

  // Parse command line arguments.
  // It is recommended that n is divisible by split, otherwise there will be
  // some unusued elements near the end.
  int ch;
  while ((ch = getopt(argc, argv, "n:s:")) != -1) {
    switch (ch) {
      case 'n':
        n = atoi(optarg);
        break;
      case 's':
        split = atoi(optarg);
        break;
      default:
        abort();
    }
  }

  // How much data will each kernel process?
  int chunk = n / split;

  printf("n: %d, split: %d, chunk: %d\n", n, split, chunk);

  if (n % split != 0) {
    printf("Warning: recommend n to be divisible by split\n");
  }

  double* h_a; double* h_b; double* h_c;
  double* d_a; double* d_b; double* d_c;

  // Allocate memory
  cudaMallocHost(&h_a, n * sizeof(double));
  cudaMallocHost(&h_b, n * sizeof(double));
  cudaMallocHost(&h_c, n * sizeof(double));
  cudaMalloc(&d_a, n * sizeof(double));
  cudaMalloc(&d_b, n * sizeof(double));
  cudaMalloc(&d_c, n * sizeof(double));

  // Initialize vectors
  for (int i = 0; i < n; i++) {
    h_a[i] = (double)i;
    h_b[i] = (double)(2*i);
  }

  // Copy vectors to GPU
  cudaMemcpy(d_a, h_a, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice);

  // Create as many streams as split
  cudaStream_t streams[split];
  for (int i = 0; i < split; i++) {
    cudaStreamCreate(&streams[i]);
  }

  dim3 block_dim(BLOCK_SIZE);
  dim3 grid_dim((chunk + block_dim.x - 1) / block_dim.x);

#if WARMUP
  // Warm up GPU
  for (int i = 0; i < WARMUP_COUNT; i++) {
    for (int j = 0; j < split; j++) {
      warmup<<<grid_dim, block_dim, 0, streams[j]>>>(d_a + chunk * j, d_b + chunk * j, chunk);
    }
    cudaDeviceSynchronize();
  }
#endif

  auto timer_start = std::chrono::system_clock::now();

  // Execute split kernels
  for (int i = 0; i < split; i++) {
    vecadd<<<grid_dim, block_dim, 0, streams[i]>>>(d_a + chunk * i, d_b + chunk * i, d_c + chunk * i, chunk);
  }
  cudaDeviceSynchronize();

  auto timer_end = std::chrono::system_clock::now();

  // Copy result vector back to host
  cudaMemcpy(h_c, d_c, n * sizeof(double), cudaMemcpyDeviceToHost);

#if VERIFY
  // Verify results
  printf("\n");
  for (int i = 0; i < n; i++) {
    printf("%.0lf ", h_c[i]);
  }
  printf("\n");
#endif

  // Print timer
  std::chrono::duration<double> elapsed = timer_end - timer_start;
  printf("\nSplit kernel execution: %.3lf us\n", elapsed.count() * 1000000);

  return 0;
}
