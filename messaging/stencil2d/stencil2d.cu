#include "hapi.h"
#include "stencil2d.h"

#define TILE_SIZE 16

__global__ void packingKernel(double* temperature, double* west_ghost,
    double* east_ghost, double* north_ghost, double* south_ghost, int block_x,
    int block_y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_y && west_ghost != NULL) {
    west_ghost[i] = temperature[(block_x + 2) * (1 + i) + 1];
  }
  else if (i < 2 * block_y && east_ghost != NULL) {
    i -= block_y;
    east_ghost[i] = temperature[(block_x + 2) * (1 + i) + block_x];
  }
  else if (i < 2 * block_y + block_x && north_ghost != NULL) {
    i -= 2 * block_y;
    north_ghost[i] = temperature[(block_x + 2) + (1 + i)];
  }
  else if (i < 2 * block_y + 2 * block_x && south_ghost != NULL) {
    i -= (2 * block_y + block_x);
    south_ghost[i] = temperature[(block_x + 2) * block_y + (1 + i)];
  }
}

__global__ void unpackingKernel(double* temperature, double* ghost, int width,
    int dir, int block_x, int block_y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < width) {
    if (dir == WEST) {
      temperature[(block_x + 2) * (1 + i)] = ghost[i];
    }
    else if (dir == EAST) {
      temperature[(block_x + 2) * (1 + i) + (block_x + 1)] = ghost[i];
    }
    else if (dir == NORTH) {
      temperature[1 + i] = ghost[i];
    }
    else if (dir == SOUTH) {
      temperature[(block_x + 2) * (block_y + 1) + (1 + i)] = ghost[i];
    }
  }
}

__global__ void stencilKernel(double* temperature, double* new_temperature,
    int block_x, int block_y, int thread_size) {
  int i_start = (blockDim.x * blockIdx.x + threadIdx.x) * thread_size + 1;
  int i_finish =
      (blockDim.x * blockIdx.x + threadIdx.x) * thread_size + thread_size;
  int j_start = (blockDim.y * blockIdx.y + threadIdx.y) * thread_size + 1;
  int j_finish =
      (blockDim.y * blockIdx.y + threadIdx.y) * thread_size + thread_size;

  for (int i = i_start; i <= i_finish; i++) {
    for (int j = j_start; j <= j_finish; j++) {
      if (i <= block_x && j <= block_y) {
        new_temperature[j * (block_x + 2) + i] =
            (temperature[j * (block_x + 2) + (i - 1)] +
             temperature[j * (block_x + 2) + (i + 1)] +
             temperature[(j - 1) * (block_x + 2) + i] +
             temperature[(j + 1) * (block_x + 2) + i] +
             temperature[j * (block_x + 2) + i]) *
            DIVIDEBY5;
      }
    }
  }
}

void invokePackingKernel(double* temperature, double* west_ghost, double* east_ghost,
    double* north_ghost, double* south_ghost, int block_x, int block_y,
    cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE * TILE_SIZE);
  dim3 grid_dim((2 * block_x + 2 * block_y + block_dim.x - 1) / block_dim.x);

  packingKernel<<<grid_dim, block_dim, 0, stream>>>(temperature, west_ghost,
      east_ghost, north_ghost, south_ghost, block_x, block_y);

  hapiCheck(cudaPeekAtLastError());
}

void invokeUnpackingKernel(double* temperature, double* ghost, int width,
    int dir, int block_x, int block_y, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE);
  dim3 grid_dim((width + block_dim.x - 1) / block_dim.x);

  unpackingKernel<<<grid_dim, block_dim, 0, stream>>>(temperature, ghost, width,
      dir, block_x, block_y);

  hapiCheck(cudaPeekAtLastError());
}

void invokeStencilKernel(double* d_temperature, double* d_new_temperature,
    int block_x, int block_y, int thread_size, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(
      (block_x + (block_dim.x * thread_size - 1)) / (block_dim.x * thread_size),
      (block_y + (block_dim.y * thread_size - 1)) / (block_dim.y * thread_size));

  stencilKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
      d_new_temperature, block_x, block_y, thread_size);

  hapiCheck(cudaPeekAtLastError());
}
