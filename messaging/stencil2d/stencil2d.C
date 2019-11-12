#include "stencil2d.decl.h"
#include <string>
#include "hapi.h"
#ifdef USE_NVTX
#include "hapi_nvtx.h"
#endif

#define PRINT 1

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

extern void invokeKernel(double* d_temperature, double* d_new_temperature,
    int block_x, int block_y, int thread_coarsening, cudaStream_t stream);

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
          CkAbort("Unknown command line argument detected");
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

    block_proxy.init();
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

  Block() {}

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
    if (thisIndex.y == 0) north_bound = true;
    else neighbors++;
    if (thisIndex.y == n_chares_y - 1) south_bound = true;
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

    // Initialize temperature data
    for (int j = 1; j <= block_y; j++) {
      for (int i = 1; i <= block_x; i++) {
        temperature[(block_x + 2) * j + i] = thisFlatIndex;
      }
    }

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

  void prepareGhosts() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::prepareGhosts",
        NVTXColor::MidnightBlue);
#endif

    // Enforce boundary conditions
    constrainBC();

    // Set up callback to be invoked once preparation is done
    CkCallback* cb = new CkCallback(CkIndex_Block::prepareGhostsDone(NULL), thisProxy[thisIndex]);
    cb->setRefnum(my_iter);

    // Copy ghost data into contiguous buffers.
    // Explicit data transfers are required without unified memory.
    if (!unified_memory) {
      if (!west_bound) {
        hapiCheck(cudaMemcpy2DAsync(west_ghost, sizeof(double),
              d_temperature + (block_x + 2) + 1,
              (block_x + 2) * sizeof(double), sizeof(double),
              block_y, cudaMemcpyDeviceToHost, stream));
      }
      if (!east_bound) {
        hapiCheck(
            cudaMemcpy2DAsync(east_ghost, sizeof(double),
              d_temperature + (block_x + 2) + block_x,
              (block_x + 2) * sizeof(double), sizeof(double),
              block_y, cudaMemcpyDeviceToHost, stream));
      }
      if (!north_bound) {
        hapiCheck(cudaMemcpyAsync(north_ghost, d_temperature + (block_x + 2) + 1,
              block_x * sizeof(double), cudaMemcpyDeviceToHost, stream));
      }
      if (!south_bound) {
        hapiCheck(cudaMemcpyAsync(south_ghost, d_temperature + (block_x + 2) * block_y + 1,
              block_x * sizeof(double), cudaMemcpyDeviceToHost, stream));
      }

      hapiAddCallback(stream, cb);
    }
    else {
      if (!west_bound) {
        for (int j = 0; j < block_y; j++) {
          west_ghost[j] = temperature[(block_x + 2) * (1 + j) + 1];
        }
      }
      if (!east_bound) {
        for (int j = 0; j < block_y; j++) {
          east_ghost[j] = temperature[(block_x + 2) * (1 + j) + block_x];
        }
      }
      if (!north_bound) {
        for (int i = 0; i < block_x; i++) {
          north_ghost[i] = temperature[(block_x + 2) + (1 + i)];
        }
      }
      if (!south_bound) {
        for (int i = 0; i < block_x; i++) {
          south_ghost[i] = temperature[(block_x + 2) * block_y + (1 + i)];
        }
      }

      cb->send();
    }
  }

  void sendGhosts(void) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::sendGhosts",
        NVTXColor::MidnightBlue);
#endif

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
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::processGhosts",
        NVTXColor::WetAsphalt);
#endif

    switch (dir) {
      case WEST:
        if (!unified_memory) {
          memcpy(west_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpy2DAsync(
              d_temperature + (block_x + 2), (block_x + 2) * sizeof(double),
              west_ghost, sizeof(double), sizeof(double), block_y,
              cudaMemcpyHostToDevice, stream));
        }
        else {
          for (int j = 0; j < width; j++) {
            temperature[(block_x + 2) * (1 + j)] = gh[j];
          }
        }
        break;
      case EAST:
        if (!unified_memory) {
          memcpy(east_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpy2DAsync(
              d_temperature + (block_x + 2) + (block_x + 1), (block_x + 2) * sizeof(double),
              east_ghost, sizeof(double), sizeof(double), block_y,
              cudaMemcpyHostToDevice, stream));
        }
        else {
          for (int j = 0; j < width; j++) {
            temperature[(block_x + 2) * (1 + j) + (block_x + 1)] = gh[j];
          }
        }
        break;
      case NORTH:
        if (!unified_memory) {
          memcpy(north_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpyAsync(d_temperature + 1,
                north_ghost, block_x * sizeof(double), cudaMemcpyHostToDevice, stream));
        }
        else {
          for (int i = 0; i < width; i++) {
            temperature[1 + i] = gh[i];
          }
        }
        break;
      case SOUTH:
        if (!unified_memory) {
          memcpy(south_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpyAsync(d_temperature + (block_x + 2) * (block_y + 1) + 1,
                south_ghost, block_x * sizeof(double), cudaMemcpyHostToDevice, stream));
        }
        else {
          for (int i = 0; i < width; i++) {
            temperature[(block_x + 2) * (block_y + 1) + (1 + i)] = gh[i];
          }
        }
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  void update() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::update",
        NVTXColor::Amethyst);
#endif

    // Set up callback
    CkCallback* cb = new CkCallback(CkIndex_Block::updateDone(NULL), thisProxy[thisIndex]);
    cb->setRefnum(my_iter);

    // Invoke Kernel
    // FIXME: Does not need to wait until data transfer is complete,
    // but what about unified memory?
    if (!unified_memory) {
      invokeKernel(d_temperature, d_new_temperature, block_x, block_y,
          thread_coarsening, stream);
    }
    else {
      invokeKernel(temperature, new_temperature, block_x, block_y,
          thread_coarsening, stream);
    }

    // Copy final temperature data back to host (on last iteration)
    if (my_iter == n_iters-1) {
      if (!unified_memory) {
        hapiCheck(cudaMemcpyAsync(temperature, d_new_temperature,
              sizeof(double) * (block_x + 2) * (block_y + 2),
              cudaMemcpyDeviceToHost, stream));
      }
    }

    hapiAddCallback(stream, cb);
  }

  void validateAndTerminate() {
#if PRINT
    CkPrintf("Block (%d, %d)\n", thisIndex.x, thisIndex.y);
    for (int j = 0; j < block_y + 2; j++) {
      for (int i = 0; i < block_x + 2; i++) {
        if (!unified_memory) {
          CkPrintf("%.3lf ", temperature[(block_x + 2) * j + i]);
        }
        else {
          CkPrintf("%.3lf ", new_temperature[(block_x + 2) * j + i]);
        }
      }
      CkPrintf("\n");
    }
#endif

    CkPrintf("[%4d] Average time per iteration: %.3lf us\n", thisFlatIndex,
        (total_time / n_iters) * 1000000);

    // Move on to next chare or terminate at last chare
    if (thisIndex.x == n_chares_x - 1 && thisIndex.y == n_chares_y - 1) {
      main_proxy.done();
    }
    else {
      if (thisIndex.x == n_chares_x - 1) {
        thisProxy(0, thisIndex.y + 1).validateAndTerminate();
      }
      else {
        thisProxy(thisIndex.x + 1, thisIndex.y).validateAndTerminate();
      }
    }
  }

  void constrainBC() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Block::constrainBC", NVTXColor::Carrot);
#endif

    if (west_bound) {
      for (int j = 1; j <= block_y; j++) {
        temperature[j * (block_x + 2)] = 1.0;
      }
    }
    if (east_bound) {
      for (int j = 1; j <= block_y; j++) {
        temperature[j * (block_x + 2) + (block_x + 1)] = 1.0;
      }
    }
    if (north_bound) {
      for (int i = 1; i <= block_x; i++) {
        temperature[i] = 1.0;
      }
    }
    if (south_bound) {
      for (int i = 1; i <= block_x; i++) {
        temperature[(block_x + 2) * (block_y + 1) + i] = 1.0;
      }
    }
  }
};

#include "stencil2d.def.h"
