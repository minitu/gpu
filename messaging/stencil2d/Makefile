OPTS = -O3 -DUSE_NVTX

CHARM_DIR = $(HOME)/charm
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

CUDA_DIR ?= $(CUDA_HOME)
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -O3 -c -std=c++11 -use_fast_math
LD_LIBS = -lnvToolsExt

TARGET = stencil2d
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu $(TARGET).h
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
