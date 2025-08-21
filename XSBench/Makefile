#===============================================================================
# User Options
#===============================================================================

CUDA_PATH ?= /usr/local/cuda-12.9

ENZYME_PATH ?= $(HOME)/git/Reactant/enzyme/buildRelease/Enzyme/ClangEnzyme-22.so
CLANG_PATH ?= $(HOME)/git/Enzyme/llvm-project/buildRelease/bin/clang++
LIB_RAISE_PATH ?= $(HOME)/git/Enzyme-JaX/bazel-bin/libRaise.so
REACTANT_EXTRA_PATH ?= $(HOME)/git/Reactant.jl/deps/ReactantExtra/bazel-bin/

export CUDA_HOME := $(CUDA_PATH)
export PATH := $(CUDA_PATH)/bin:$(PATH)
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE = 128
export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)

SM_VERSION  = 120


OPTIMIZE    ?= yes
FORWARD     ?= yes
INLINE      ?= yes
NEWCACHE    ?= yes
MINCACHE    ?= yes
AA          ?= yes
PHISTRUCT   ?= yes
TEMPLATIZE  ?= no

COALESE     ?= yes

CACHELICM   ?= yes

SELECT      ?= no
SPECPHI     ?= no

VERIFY      ?= no

DEBUG       = no
PROFILE     = no

DYN ?= yes

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.cu \
io.cu \
Simulation.cu \
GridInit.cu \
XSutils.cu \
Materials.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================


# Linker Flags
LDFLAGS = -lcudart_static "-ldl" "-lrt" -lpthread -lm -L$(REACTANT_EXTRA_PATH) -lReactantExtra

CC := $(CLANG_PATH)
CFLAGS += -ffast-math --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11 -Xclang -load -Xclang $(ENZYME_PATH) -Xclang -add-plugin -Xclang enzyme -mllvm -raising-plugin-path -mllvm $(LIB_RAISE_PATH) -mllvm -reactant-backend -mllvm cuda -x cuda -fno-exceptions

#CFLAGS += -mllvm -simplifycfg-sink-common=0

# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g -G
	LDFLAGS  += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
	CFLAGS += -pg
	LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif

# Optimization Flags
ifeq ($(FORWARD),yes)
	CFLAGS += -DFORWARD_PASS=1
endif

# Optimization Flags
ifeq ($(INLINE),yes)
	CFLAGS += -DALWAYS_INLINE=1
endif

# Optimization Flags
# ifeq ($(NEWCACHE),yes)
# 	CFLAGS += -mllvm -enzyme-new-cache=1
# else
# 	CFLAGS += -mllvm -enzyme-new-cache=0
# endif


# ifeq ($(DYN),yes)
#   CFLAGS += -mllvm -enzyme-inactive-dynamic=1
# else
#   CFLAGS += -mllvm -enzyme-inactive-dynamic=0
# endif

# ifeq ($(MINCACHE),yes)
#   CFLAGS += -mllvm -enzyme-mincut-cache=1
# else
#   CFLAGS += -mllvm -enzyme-mincut-cache=0
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

# ifeq ($(CACHELICM),yes)
#   CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=1
# else
#   CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=0
# endif

# ifeq ($(AA),yes)
#   CFLAGS += -mllvm -enzyme-aggressive-aa=1
# else
#   CFLAGS += -mllvm -enzyme-aggressive-aa=0
# endif

# ifeq ($(COALESE),yes)
#   CFLAGS += -mllvm -enzyme-coalese
# endif

# ifeq ($(PHISTRUCT),yes)
#   CFLAGS += -mllvm -enzyme-phi-restructure=1
# else
#   CFLAGS += -mllvm -enzyme-phi-restructure=0
# endif


# ifeq ($(TEMPLATIZE),yes)
#   CFLAGS += -DTEMPLATIZE=1
# endif

# ifeq ($(VERIFY),yes)
#   CFLAGS += -DPRINT=1
# endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(source) Makefile
	$(CC) $(CFLAGS) $(source) $(LDFLAGS) -o $@

# $(program): $(obj) XSbench_header.cuh Makefile
# 	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

# %.o: %.cu XSbench_header.cuh Makefile
# 	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.re_export

edit:
	vim -p $(source) XSbench_header.cuh

run:
	./$(program)
