#include "stencil2d.decl.h"
#include <string>
#include "hapi.h"
#ifdef USE_NVTX
#include "hapi_nvtx.h"
#endif
#include "stencil2d.h"

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_GPUHandler gpuhandler_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int grid_dim;
/* readonly */ int block_x;
/* readonly */ int block_y;
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
/* readonly */ int n_iters;
/* readonly */ int thread_coarsening;
/* readonly */ bool unified_memory;
/* readonly */ bool print_block;

extern void invokeInitKernel(double* temperature, double val, int block_x,
    int block_y, int thread_coarsening, cudaStream_t stream);
extern void invokePackingKernel(double* temperature, double* west_ghost,
    double* east_ghost, double* north_ghost, double* south_ghost, int block_x,
    int block_y, cudaStream_t stream);
extern void invokeUnpackingKernel(double* temperature, double* ghost, int width,
    int dir, int block_x, int block_y, cudaStream_t stream);
extern void invokeBoundaryKernel(double* temperature, bool west_bound,
    bool east_bound, bool north_bound, bool south_bound, int block_x, int block_y,
    cudaStream_t stream);
extern void invokeStencilKernel(double* d_temperature, double* d_new_temperature,
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
    print_block = false;

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:y:i:t:up")) != -1) {
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
        case 'p':
          print_block = true;
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

    // Override GPU settings set by HAPI
    gpuhandler_proxy = CProxy_GPUHandler::ckNew();
    gpuhandler_proxy.setGPU();
  }

  void ready() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::ready", NVTXColor::Turquoise);
#endif

    // Create 2D chare array
    block_proxy = CProxy_Block::ckNew(n_chares_x, n_chares_y);
    start_time = CkWallTimer();
    block_proxy.init();
  }

  void finalize() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::finalize", NVTXColor::Turquoise);
#endif

    block_proxy(0, 0).validate();
  }

  void terminate() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::terminate", NVTXColor::Turquoise);
#endif

    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class GPUHandler : public CBase_GPUHandler {
public:
  int device_count;
  int local_pe_id;
  int gpu_id;

  GPUHandler() {
    device_count = 0;
    gpu_id = 0;
  }

  void setGPU() {
    hapiCheck(cudaGetDeviceCount(&device_count));
    CkAssert(device_count > 0);

    // Block mapping of PEs to GPUs
    int pes_per_process = CkNumPes() / CkNumNodes();
    local_pe_id = CkMyPe() % pes_per_process;
    gpu_id = local_pe_id / (pes_per_process / device_count);
    hapiCheck(cudaSetDevice(gpu_id));

    CkPrintf("[PE %d, LPE %d] Set CUDA device to %d\n", CkMyPe(), local_pe_id, gpu_id);

    contribute(CkCallback(CkReductionTarget(Main, ready), main_proxy));
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
  double* h_temperature;
  double* h_new_temperature;
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

    if (unified_memory) {
      hapiCheck(cudaFree(temperature));
      hapiCheck(cudaFree(new_temperature));
      hapiCheck(cudaFree(west_ghost));
      hapiCheck(cudaFree(east_ghost));
      hapiCheck(cudaFree(north_ghost));
      hapiCheck(cudaFree(south_ghost));
    }
    else {
      hapiCheck(cudaFree(temperature));
      hapiCheck(cudaFree(new_temperature));
      hapiCheck(cudaFreeHost(h_temperature));
      hapiCheck(cudaFreeHost(h_new_temperature));
      hapiCheck(cudaFreeHost(west_ghost));
      hapiCheck(cudaFreeHost(east_ghost));
      hapiCheck(cudaFreeHost(north_ghost));
      hapiCheck(cudaFreeHost(south_ghost));
    }

    cudaStreamDestroy(stream);
  }

  void init() {
    thisFlatIndex = n_chares_x * thisIndex.y + thisIndex.x;

#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::init",
        NVTXColor::PeterRiver);
#endif

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
    if (unified_memory) {
      hapiCheck(cudaMallocManaged(&temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMallocManaged(&new_temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMallocManaged(&west_ghost, sizeof(double) * block_y));
      hapiCheck(cudaMallocManaged(&east_ghost, sizeof(double) * block_y));
      hapiCheck(cudaMallocManaged(&south_ghost, sizeof(double) * block_x));
      hapiCheck(cudaMallocManaged(&north_ghost, sizeof(double) * block_x));
    }
    else {
      hapiCheck(cudaMalloc(&temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMalloc(&new_temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMallocHost(&h_temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMallocHost(&h_new_temperature, sizeof(double) * (block_x+2) * (block_y+2)));
      hapiCheck(cudaMallocHost(&west_ghost, sizeof(double) * block_y));
      hapiCheck(cudaMallocHost(&east_ghost, sizeof(double) * block_y));
      hapiCheck(cudaMallocHost(&south_ghost, sizeof(double) * block_x));
      hapiCheck(cudaMallocHost(&north_ghost, sizeof(double) * block_x));
    }

    cudaStreamCreate(&stream);

    // Initialize temperature data
    invokeInitKernel(temperature, (double)thisFlatIndex, block_x, block_y,
        thread_coarsening, stream);

    CkCallback* cb = new CkCallback(CkIndex_Block::iterate(), thisProxy[thisIndex]);
    hapiAddCallback(stream, cb);
  }

  void prepareGhosts() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Block::prepareGhosts",
        NVTXColor::MidnightBlue);
#endif

    // Set up callback to be invoked once preparation is done
    CkCallback* cb = new CkCallback(CkIndex_Block::prepareGhostsDone(NULL), thisProxy[thisIndex]);
    cb->setRefnum(my_iter);

    // Copy ghost data into contiguous buffers.
    // Explicit data transfers are required without unified memory.
    if (unified_memory) {
      invokePackingKernel(temperature, west_bound ? NULL : west_ghost,
          east_bound ? NULL : east_ghost, north_bound ? NULL : north_ghost,
          south_bound ? NULL : south_ghost, block_x, block_y, stream);
    }
    else {
      if (!west_bound) {
        hapiCheck(cudaMemcpy2DAsync(west_ghost, sizeof(double),
              temperature + (block_x + 2) + 1,
              (block_x + 2) * sizeof(double), sizeof(double),
              block_y, cudaMemcpyDeviceToHost, stream));
      }
      if (!east_bound) {
        hapiCheck(
            cudaMemcpy2DAsync(east_ghost, sizeof(double),
              temperature + (block_x + 2) + block_x,
              (block_x + 2) * sizeof(double), sizeof(double),
              block_y, cudaMemcpyDeviceToHost, stream));
      }
      if (!north_bound) {
        hapiCheck(cudaMemcpyAsync(north_ghost, temperature + (block_x + 2) + 1,
              block_x * sizeof(double), cudaMemcpyDeviceToHost, stream));
      }
      if (!south_bound) {
        hapiCheck(cudaMemcpyAsync(south_ghost, temperature + (block_x + 2) * block_y + 1,
              block_x * sizeof(double), cudaMemcpyDeviceToHost, stream));
      }
    }

    // Set to proceed once previous GPU operations are complete
    hapiAddCallback(stream, cb);
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
        memcpy(west_ghost, gh, width * sizeof(double));
        if (unified_memory) {
          invokeUnpackingKernel(temperature, west_ghost, width, WEST, block_x,
              block_y, stream);
        }
        else {
          hapiCheck(cudaMemcpy2DAsync(
              temperature + (block_x + 2), (block_x + 2) * sizeof(double),
              west_ghost, sizeof(double), sizeof(double), block_y,
              cudaMemcpyHostToDevice, stream));
        }
        break;
      case EAST:
        memcpy(east_ghost, gh, width * sizeof(double));
        if (unified_memory) {
          invokeUnpackingKernel(temperature, east_ghost, width, EAST, block_x,
              block_y, stream);
        }
        else {
          hapiCheck(cudaMemcpy2DAsync(
              temperature + (block_x + 2) + (block_x + 1), (block_x + 2) * sizeof(double),
              east_ghost, sizeof(double), sizeof(double), block_y,
              cudaMemcpyHostToDevice, stream));
        }
        break;
      case NORTH:
        memcpy(north_ghost, gh, width * sizeof(double));
        if (unified_memory) {
          invokeUnpackingKernel(temperature, north_ghost, width, NORTH, block_x,
              block_y, stream);
        }
        else {
          hapiCheck(cudaMemcpyAsync(temperature + 1, north_ghost,
                block_x * sizeof(double), cudaMemcpyHostToDevice, stream));
        }
        break;
      case SOUTH:
        memcpy(south_ghost, gh, width * sizeof(double));
        if (unified_memory) {
          invokeUnpackingKernel(temperature, south_ghost, width, SOUTH, block_x,
              block_y, stream);
        }
        else {
          hapiCheck(cudaMemcpyAsync(temperature + (block_x + 2) * (block_y + 1) + 1,
                south_ghost, block_x * sizeof(double), cudaMemcpyHostToDevice, stream));
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

    // Set boundary conditions for ghost regions that have not received data
    // from neighbors
    invokeBoundaryKernel(temperature, west_bound, east_bound, north_bound,
        south_bound, block_x, block_y, stream);

    // Execute stencil updates
    invokeStencilKernel(temperature, new_temperature, block_x, block_y,
        thread_coarsening, stream);

    // Copy final temperature data back to host (on last iteration)
    if (my_iter == n_iters-1 && !unified_memory) {
      hapiCheck(cudaMemcpyAsync(h_temperature, temperature,
            sizeof(double) * (block_x + 2) * (block_y + 2),
            cudaMemcpyDeviceToHost, stream));
      hapiCheck(cudaMemcpyAsync(h_new_temperature, new_temperature,
            sizeof(double) * (block_x + 2) * (block_y + 2),
            cudaMemcpyDeviceToHost, stream));
    }

    // Set up callback
    CkCallback* cb = new CkCallback(CkIndex_Block::updateDone(NULL), thisProxy[thisIndex]);
    cb->setRefnum(my_iter);
    hapiAddCallback(stream, cb);
  }

  void validate() {
    if (print_block) {
      CkPrintf("Block (%d, %d)\n", thisIndex.x, thisIndex.y);

      // Old temperature data
      CkPrintf("Old:\n");
      double* temperature_val = unified_memory ? temperature : h_temperature;
      for (int j = 0; j < block_y + 2; j++) {
        for (int i = 0; i < block_x + 2; i++) {
          CkPrintf("%.3lf ", temperature_val[(block_x + 2) * j + i]);
        }
        CkPrintf("\n");
      }

      // New temperature data
      temperature_val = unified_memory ? new_temperature : h_new_temperature;
      CkPrintf("New:\n");
      for (int j = 0; j < block_y + 2; j++) {
        for (int i = 0; i < block_x + 2; i++) {
          CkPrintf("%.3lf ", temperature_val[(block_x + 2) * j + i]);
        }
        CkPrintf("\n");
      }
    }

    CkPrintf("[%4d] Average time per iteration: %.3lf us\n", thisFlatIndex,
        (total_time / n_iters) * 1000000);

    // Move on to next chare or terminate at last chare
    if (thisIndex.x == n_chares_x - 1 && thisIndex.y == n_chares_y - 1) {
      main_proxy.terminate();
    }
    else {
      if (thisIndex.x == n_chares_x - 1) {
        thisProxy(0, thisIndex.y + 1).validate();
      }
      else {
        thisProxy(thisIndex.x + 1, thisIndex.y).validate();
      }
    }
  }
};

#include "stencil2d.def.h"
