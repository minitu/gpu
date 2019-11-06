#include "vecadd.decl.h"
#include "hapi.h"
#ifdef USE_NVTX
#include "hapi_nvtx.h"
#endif

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int n_chares;
/* readonly */ int vector_size;
/* readonly */ int split;
/* readonly */ int chunk;

extern void cudaVecAdd(int, double*, double*, double*, double*, double*, double*,
                       cudaStream_t);

class Main : public CBase_Main {
 private:
  CProxy_Chunk chunks;
  double start_time;

 public:
  Main(CkArgMsg* m) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::Main", NVTXColor::Turquoise);
#endif

    mainProxy = thisProxy;
    n_chares = 4;
    vector_size = 1024;
    split = 1;

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "c:n:s:")) != -1) {
      switch (c) {
        case 'c':
          n_chares = atoi(optarg);
          break;
        case 'n':
          vector_size = atoi(optarg);
          break;
        case 's':
          split = atoi(optarg);
          break;
        default:
          CkPrintf("Usage: %s -c [chares] -n [vector size] -s [split]\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // Data size per chare (PE)
    chunk = vector_size / split;

    CkPrintf("chares: %d, vector size: %d, split: %d, chunk: %d\n", n_chares,
        vector_size, split, chunk);

    if (vector_size % split != 0) {
      CkAbort("Vector size should be divisible by split");
    }

    start_time = CkWallTimer();

    // Create chunk chares and initiate H2D data transfers
    chunks = CProxy_Chunk::ckNew(n_chares);
    chunks.transfer();
  }

  void transfer_complete() {
#ifdef USE_NVTXX
    NVTXTracer nvtx_range("Main::transfer_complete", NVTXColor::Turquoise);
#endif

    chunks.kernel();
  }

  void all_complete() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::all_complete", NVTXColor::Turquoise);
#endif

    CkPrintf("\nElapsed time: %.3lf us\n", (CkWallTimer() - start_time) * 1000000);
    CkExit();
  }
};

class Chunk : public CBase_Chunk {
 private:
  double* h_A;
  double* h_B;
  double* h_C;
  double* d_A;
  double* d_B;
  double* d_C;
  size_t size;
  cudaStream_t stream;

 public:
  Chunk() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Chunk::Chunk", NVTXColor::WetAsphalt);
#endif

    size = chunk * sizeof(double);
    hapiCheck(cudaMallocHost(&h_A, size));
    hapiCheck(cudaMallocHost(&h_B, size));
    hapiCheck(cudaMallocHost(&h_C, size));
    hapiCheck(cudaMalloc(&d_A, size));
    hapiCheck(cudaMalloc(&d_B, size));
    hapiCheck(cudaMalloc(&d_C, size));
    hapiCheck(cudaStreamCreate(&stream));

    for (int i = 0; i < chunk; i++) {
      h_A[i] = (double)i;
      h_B[i] = (double)i;
    }
  }

  ~Chunk() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Chunk::~Chunk", NVTXColor::WetAsphalt);
#endif

    hapiCheck(cudaFreeHost(h_A));
    hapiCheck(cudaFreeHost(h_B));
    hapiCheck(cudaFreeHost(h_C));
    hapiCheck(cudaFree(d_A));
    hapiCheck(cudaFree(d_B));
    hapiCheck(cudaFree(d_C));
    hapiCheck(cudaStreamDestroy(stream));
  }

  void transfer() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Chunk::transfer", NVTXColor::Carrot);
#endif

    // Copy input vectors to device
    hapiCheck(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    hapiCheck(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));
    hapiCheck(cudaStreamSynchronize(stream));

    // Synchronize
    contribute(CkCallback(CkReductionTarget(Main, transfer_complete), mainProxy));
  }

  void kernel() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Chunk::kernel", NVTXColor::Carrot);
#endif

    // Invoke kernel
    cudaVecAdd(chunk, h_A, h_B, h_C, d_A, d_B, d_C, stream);

    // Copy output vector to host
    hapiCheck(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

    // Set up callback
    CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
    CkCallback* cb =
        new CkCallback(CkIndex_Chunk::complete(), myIndex, thisArrayID);
    hapiAddCallback(stream, cb);
  }

  void complete() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Chunk::complete", NVTXColor::Clouds);
#endif

    contribute(CkCallback(CkReductionTarget(Main, all_complete), mainProxy));
  }
};

#include "vecadd.def.h"
