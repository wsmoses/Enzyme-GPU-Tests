# Environment Variables

CUDA_PATH ?= /usr/local/cuda-12.9

EMBEDDED_CLANG ?= no

ENZYME_PATH ?= $(HOME)/git/Reactant/enzyme/buildRelease/Enzyme/ClangEnzyme-22.so
CLANG_PATH ?= $(HOME)/git/Enzyme/llvm-project/buildRelease/bin/clang++
LIB_RAISE_PATH ?= $(HOME)/git/Enzyme-JaX/bazel-bin/libRaise.so
REACTANT_EXTRA_PATH ?= $(HOME)/git/Reactant.jl/deps/ReactantExtra/bazel-bin/

export CUDA_HOME := $(CUDA_PATH)
export PATH := $(CUDA_PATH)/bin:$(PATH)
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE = 128
export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)

SM_VERSION  = 120

# Ablation testing & numerical verification
VERIFY  ?= no
FORWARD ?= yes
PHIOPT ?= no
BRANCHYOPT ?= no
MINCCACHE ?= yes
NEWCACHE ?= yes
OPTIMIZE ?= yes
RESTRICT ?= yes

SPECPHI ?= no
SELECT ?= no

DEBUG ?= no

CC := $(CLANG_PATH)

ENZYME_LOAD_FLAGS :=
ifeq ($(EMBEDDED_CLANG),no)
  ENZYME_LOAD_FLAGS += -fplugin=$(ENZYME_PATH) -Xclang -add-plugin -Xclang enzyme -mllvm -raising-plugin-path -mllvm $(LIB_RAISE_PATH)
endif

CFLAGS  = -DNDEBUG -I. -ffast-math --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION) -std=c++11 $(ENZYME_LOAD_FLAGS) -mllvm -reactant-backend -mllvm cuda -x cuda -fno-exceptions
LDFLAGS = -lcudart_static "-ldl" "-lrt" -lpthread -lm -L$(REACTANT_EXTRA_PATH)

source = allocator.cu lulesh-comms.cu lulesh-comms-gpu.cu lulesh.cu

ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3 
endif

ifeq ($(RESTRICT),yes)
	CFLAGS += -DRESTRICT=1
else
	CFLAGS += -DRESTRICT=0
endif

ifeq ($(DEBUG),yes)
	CFLAGS += -g
endif

ifeq ($(VERIFY),yes)
	CFLAGS += -DVERIFY
endif

# ifeq ($(MINCCACHE),yes)
# 	CFLAGS += -mllvm -enzyme-mincut-cache=1
# else
# 	CFLAGS += -mllvm -enzyme-mincut-cache=0
# endif

# ifeq ($(NEWCACHE),yes)
# 	CFLAGS += -mllvm -enzyme-new-cache=1
# else
# 	CFLAGS += -mllvm -enzyme-new-cache=0
# endif

# ifeq ($(SELECT),yes)
#   CFLAGS += -mllvm -enzyme-select-opt=1
# else
#   CFLAGS += -mllvm -enzyme-select-opt=0
# endif

# ifeq ($(SPECPHI),yes)
#   CFLAGS += -mllvm -enzyme-speculate-phis=1
# else
#   CFLAGS += -mllvm -enzyme-speculate-phis=0
# endif

# ifeq ($(BRANCHYOPT),yes)
# 	CFLAGS += -DBRANCHYOPT
# endif

# ifeq ($(PHIOPT),yes)
# 	CFLAGS += -mllvm -enzyme-phi-restructure=1
# else
# 	CFLAGS += -mllvm -enzyme-phi-restructure=0
# endif

ifeq ($(FORWARD),yes)
	CFLAGS += -DNormal_forward=1
else
	CFLAGS += -DNormal_forward=0
endif
#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

#LINKFLAGS = -lmpich -L$(MPICH_DIR)/lib 
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

all: release 

debug: LINKFLAGS += 

release: 	FLAGS += $(RFLAGS)
debug: 		FLAGS += $(DFLAGS)

release: lulesh
debug: lulesh

lulesh: $(source)
	$(CC) $(CFLAGS) $(source) $(LDFLAGS) -o $@

# lulesh: allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o
# 	$(CC) $(CFLAGS) allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o -o lulesh $(LDFLAGS)

# allocator.o: allocator.cu vector.h
# 	$(CC) $(CFLAGS) allocator.cu -I ./ -c -o allocator.o

# lulesh.o: lulesh.cu util.h vector.h allocator.h
# 	$(CC) $(CFLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o lulesh.o

# lulesh-comms.o: lulesh-comms.cu
# 	$(CC) $(CFLAGS) lulesh-comms.cu -I ./ -c -o lulesh-comms.o

# lulesh-comms-gpu.o: lulesh-comms-gpu.cu
# 	$(CC) $(CFLAGS) lulesh-comms-gpu.cu -I ./ -c -o lulesh-comms-gpu.o

clean: 
	rm -rf allocator.o lulesh-comms.o lulesh-comms-gpu.o lulesh.o lulesh xyz.asc regNumList.txt *.re_export

run:
	./$(a)
