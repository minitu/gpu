#include "stencil2d.decl.h"
#include <string>
#include "hapi.h"
#ifdef USE_NVTX
#include "hapi_nvtx.h"
#endif

#define WEST 1
#define EAST 2
#define NORTH 3
#define SOUTH 4
#define DIVIDEBY5 0.2

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int grid_dim;
/* readonly */ int block_x;
/* readonly */ int block_y;
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
/* readonly */ int n_iters;
/* readonly */ int thread_coarsening;
/* readonly */ bool unified_memory;

extern void invokeKernel(cudaStream_t stream, double* d_temperature,
                         double* d_new_temperature, int block_x, int block_y,
                         int thread_coarsening);

class Main : public CBase_Main {
  double start_time;

 public:
  Main(CkArgMsg* m) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::Main", NVTXColor::Turquoise);
#endif

    main_proxy = thisProxy;
    grid_dim = 1024;
    n_chares_x = 1;
    n_chares_y = 1;
    n_iters = 100;
    thread_coarsening = 1;
    unified_memory = false;

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:y:i:t:u")) != -1) {
      switch (c) {
        case 's':
          grid_dim = atoi(optarg);
          break;
        case 'x':
          n_chares_x = atoi(optarg);
          break;
        case 'y':
          n_chares_y = atoi(optarg);
          break;
        case 'i':
          n_iters = atoi(optarg);
          break;
        case 't':
          thread_coarsening = atoi(optarg);
          break;
        case 'u':
          unified_memory = true;
          break;
        default:
          CkAbort();
      }
    }
    delete m;

    if (grid_dim % n_chares_x != 0 || grid_dim % n_chares_y != 0) {
      CkAbort("Grid indivisible by given number of chares");
    }

    block_x = grid_dim / n_chares_x;
    block_y = grid_dim / n_chares_y;

    // Print info
    CkPrintf("Grid: %d x %d, Chares: %d x %d, Block: %d x %d\n", grid_dim, grid_dim,
        n_chares_x, n_chares_y, block_x, block_y);
    CkPrintf("Iters: %d, Thread coarsening: %d, Unified memory: %d\n", n_iters,
        thread_coarsening, unified_memory);

    // Create 2D chare array
    block_proxy = CProxy_Block::ckNew(n_chares_x, n_chares_y);

    start_time = CkWallTimer();

    stencils.init();
  }

  void done() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::done", NVTXColor::Turquoise);
#endif

    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class Block : public CBase_Block {
  Block_SDAG_CODE

 public:
  int thisFlatIndex;
  int my_iter;
  int neighbors;
  int remote_count;
  double iter_start_time;
  double total_time;

  bool west_bound, east_bound, north_bound, south_bound;

  double* temperature;
  double* new_temperature;
  double* d_temperature;
  double* d_new_temperature;
  double* west_ghost;
  double* east_ghost;
  double* south_ghost;
  double* north_ghost;

  cudaStream_t stream;

  Block() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::Block",
        NVTXColor::Carrot);
#endif

    thisFlatIndex = n_chares_x * thisIndex.y + thisIndex.x;
    my_iter = 0;
    neighbors = 0;
    total_time = 0.0;

    // Check bounds and set number of valid neighbors
    west_bound = east_bound = north_bound = south_bound = false;
    if (thisIndex.x == 0) west_bound = true;
    else neighbors++;
    if (thisIndex.x == n_chares_x - 1) east_bound = true;
    else neighbors++;
    if (thisIndex.y == 0) south_bound = true;
    else neighbors++;
    if (thisIndex.y == n_chares_y - 1) north_bound = true;
    else neighbors++;

    // Allocate memory
    if (!unified_memory) {
      hapiCheck(cudaMallocHost(&temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMalloc(&d_temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMalloc(&d_new_temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMallocHost(&west_ghost, sizeof(double) * block_y));
      hapiCheck(cudaMallocHost(&east_ghost, sizeof(double) * block_y));
      hapiCheck(cudaMallocHost(&south_ghost, sizeof(double) * block_x));
      hapiCheck(cudaMallocHost(&north_ghost, sizeof(double) * block_x));
    }
    else {
      hapiCheck(cudaMallocManaged(&temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMallocManaged(&new_temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMallocManaged(&west_ghost, sizeof(double) * block_y));
      hapiCheck(cudaMallocManaged(&east_ghost, sizeof(double) * block_y));
      hapiCheck(cudaMallocManaged(&south_ghost, sizeof(double) * block_x));
      hapiCheck(cudaMallocManaged(&north_ghost, sizeof(double) * block_x));
    }

    cudaStreamCreate(&stream);
  }

  ~Block() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::~Block",
        NVTXColor::Carrot);
#endif

    if (!unified_memory) {
      hapiCheck(cudaFreeHost(temperature));
      hapiCheck(cudaFree(d_temperature));
      hapiCheck(cudaFree(d_new_temperature));
      hapiCheck(cudaFreeHost(west_ghost));
      hapiCheck(cudaFreeHost(east_ghost));
      hapiCheck(cudaFreeHost(north_ghost));
      hapiCheck(cudaFreeHost(south_ghost));
    }
    else {
      hapiCheck(cudaFree(temperature));
      hapiCheck(cudaFree(new_temperature));
      hapiCheck(cudaFree(west_ghost));
      hapiCheck(cudaFree(east_ghost));
      hapiCheck(cudaFree(north_ghost));
      hapiCheck(cudaFree(south_ghost));
    }

    cudaStreamDestroy(stream);
  }

  void init() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::init",
        NVTXColor::PeterRiver);
#endif

    // Initialize temperature data
    for (int j = 0; j < block_y + 2; j++) {
      for (int i = 0; i < block_x + 2; i++) {
        temperature[(block_x + 2) * j + i] = 0.0;
      }
    }

    // Enforce boundary conditions
    constrainBC();

    // If not using unified memory, copy initialized data to device
    if (!unified_memory) {
      hapiCheck(cudaMemcpyAsync(d_temperature, temperature,
            sizeof(double) * (block_x+2) * (block_y+2), cudaMemcpyHostToDevice, stream));

      CkCallback* cb = new CkCallback(CkIndex_Block::iterate(), thisProxy[thisIndex]);
      hapiAddCallback(stream, cb);
    }
    else {
      thisProxy[thisIndex].iterate();
    }
  }

  void sendGhosts(void) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::sendGhosts",
        NVTXColor::PeterRiver);
#endif

    // Copy ghost data into contiguous buffers
    if (!unified_memory) {
      // West ghost
      hapiCheck(cudaMemcpy2DAsync(west_ghost, sizeof(double),
            d_new_temperature + (block_x + 2) + 1,
            (block_x + 2) * sizeof(double), sizeof(double),
            block_y, cudaMemcpyDeviceToHost, stream));

      // East ghost
      hapiCheck(
          cudaMemcpy2DAsync(east_ghost, sizeof(double),
            d_new_temperature + (block_x + 2) + block_x,
            (block_x + 2) * sizeof(double), sizeof(double),
            block_y, cudaMemcpyDeviceToHost, stream));

      // North ghost
      hapiCheck(cudaMemcpyAsync(north_ghost, d_new_temperature + (block_x + 2) + 1,
            block_x * sizeof(double), cudaMemcpyDeviceToHost, stream));

      // South ghost
      hapiCheck(cudaMemcpyAsync(south_ghost, d_new_temperature + (block_x + 2) * block_y + 1,
            block_x * sizeof(double), cudaMemcpyDeviceToHost, stream));
    }
    else {
      for (int j = 0; j < block_y; j++) {
        west_ghost[j] = temperature[(block_x + 2) * (1 + j) + 1];
        east_ghost[j] = temperature[(block_x + 2) * (1 + j) + block_x];
      }

      for (int i = 0; i < block_x; i++) {
        north_ghost[i] = temperature[(block_x + 2) + (1 + i)];
        south_ghost[i] = temperature[(block_x + 2) * block_y + (1 + i)];
      }
    }

    // Send ghost data to neighbors
    int x = thisIndex.x, y = thisIndex.y;
    if (!west_bound)
      thisProxy(x - 1, y).receiveGhosts(my_iter, EAST, block_y, west_ghost);
    if (!east_bound)
      thisProxy(x + 1, y).receiveGhosts(my_iter, WEST, block_y, east_ghost);
    if (!north_bound)
      thisProxy(x, y - 1).receiveGhosts(my_iter, SOUTH, block_x, north_ghost);
    if (!south_bound)
      thisProxy(x, y + 1).receiveGhosts(my_iter, NORTH, block_x, south_ghost);
  }

  void processGhosts(int dir, int width, double* gh) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Stencil::processGhosts", NVTXColor::WetAsphalt);
#endif
    switch (dir) {
      case WEST:
        if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
          memcpy(west_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpy2DAsync(
              d_temperature + (block_x + 2), (block_x + 2) * sizeof(double),
              west_ghost, sizeof(double), sizeof(double), block_y,
              cudaMemcpyHostToDevice, stream));
        } else {
          for (int j = 0; j < width; j++) {
            temperature[(block_x + 2) * (1 + j)] = gh[j];
          }
        }
        break;
      case EAST:
        if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
          memcpy(east_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpy2DAsync(
              d_temperature + (block_x + 2) + (block_x + 1),
              (block_x + 2) * sizeof(double), east_ghost, sizeof(double),
              sizeof(double), block_y, cudaMemcpyHostToDevice, stream));
        } else {
          for (int j = 0; j < width; j++) {
            temperature[(block_x + 2) * (1 + j) + (block_x + 1)] = gh[j];
          }
        }
        break;
      case SOUTH:
        if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
          memcpy(south_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpyAsync(d_temperature + 1, south_ghost,
                                    block_x * sizeof(double),
                                    cudaMemcpyHostToDevice, stream));
        } else {
          for (int j = 0; j < width; j++) {
            temperature[1 + j] = gh[j];
          }
        }
        break;
      case NORTH:
        if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
          memcpy(north_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpyAsync(
              d_temperature + (block_x + 2) * (block_y + 1) + 1, north_ghost,
              block_x * sizeof(double), cudaMemcpyHostToDevice, stream));
        } else {
          for (int j = 0; j < width; j++) {
            temperature[(block_x + 2) * (block_y + 1) + (1 + j)] = gh[j];
          }
        }
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  // Updates local data with stencil computation.
  void update() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Stencil::update", NVTXColor::Amethyst);
#endif

    CallbackMsg* m = new CallbackMsg();
    if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
      // Invoke 2D stencil kernel
      invokeKernel(stream, d_temperature, d_new_temperature, block_x, block_y,
                   thread_coarsening);


      // Copy final temperature data back to host (on last iteration)
      if (my_iter == n_iters - 1) {
        hapiCheck(
            cudaMemcpyAsync(temperature, d_new_temperature,
                            sizeof(double) * (block_x + 2) * (block_y + 2),
                            cudaMemcpyDeviceToHost, stream));
      }

      if (local_exec_mode == CUDA_MODE) {
        cudaStreamSynchronize(stream);

        thisProxy(thisIndex.x, thisIndex.y).iterate(m);
      } else {
        CkArrayIndex2D myIndex = CkArrayIndex2D(thisIndex);
        CkCallback* cb =
            new CkCallback(CkIndex_Stencil::iterate(NULL), myIndex, thisProxy);
        if (gpu_prio)
          CkSetQueueing(m, CK_QUEUEING_LIFO);
        hapiAddCallback(stream, cb, m);
      }
    } else {  // CPU_MODE
      for (int i = 1; i <= block_x; ++i) {
        for (int j = 1; j <= block_y; ++j) {
          // Update my value based on the surrounding values
          new_temperature[j * (block_x + 2) + i] =
              (temperature[j * (block_x + 2) + (i - 1)] +
               temperature[j * (block_x + 2) + (i + 1)] +
               temperature[(j - 1) * (block_x + 2) + i] +
               temperature[(j + 1) * (block_x + 2) + i] +
               temperature[j * (block_x + 2) + i]) *
              DIVIDEBY5;
        }
      }
      double* tmp;
      tmp = temperature;
      temperature = new_temperature;
      new_temperature = tmp;

      thisProxy(thisIndex.x, thisIndex.y).iterate(m);
    }
  }

  void constrainBC() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Block::constrainBC", NVTXColor::Carrot);
#endif

    if (west_bound) {
      for (int j = 0; j < block_y + 2; ++j) {
        temperature[j * (block_x + 2)] = 1.0;
      }
    }
    if (east_bound) {
      for (int j = 0; j < block_y + 2; ++j) {
        temperature[j * (block_x + 2) + (block_x + 1)] = 1.0;
      }
    }
    if (north_bound) {
      for (int i = 0; i < block_x + 2; ++i) {
        temperature[(block_y + 1) * (block_x + 2) + i] = 1.0;
      }
    }
    if (south_bound) {
      for (int i = 0; i < block_x + 2; ++i) {
        temperature[i] = 1.0;
      }
    }
  }
};

#include "stencil2d.def.h"
