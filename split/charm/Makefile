CHARM_DIR = $(HOME)/charm
CHARMC = $(CHARM_DIR)/bin/charmc $(DEFS) $(OPTS)

# -DUSE_WR: use deprecated hapiWorkRequest for GPU offloading
# -USE_NVTX: use NVTX to profile CPU functions, need -lnvToolsExt in LD_LIBS
DEFS = #-DUSE_WR -USE_NVTX

# set CUDATOOLKIT_HOME to the CUDA toolkit directory
#NVCC := $(shell which nvcc)
CUDA_DIR = $(CUDA_HOME)
NVCC = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS = -c -std=c++11 $(DEFS)
NVCC_INC = -I$(CUDA_DIR)/include
CHARM_INC = -I$(CHARM_DIR)/include
LD_LIBS = #-lnvToolsExt

TARGET = vecadd
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(NVCC) -o $@ $(NVCC_FLAGS) $(NVCC_INC) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
