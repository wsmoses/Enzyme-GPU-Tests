#===============================================================================
# User Options
#===============================================================================

CUDA_PATH ?= /usr/local/cuda-12.9

EMBEDDED_CLANG ?= no
USE_ENZYME_LLVM ?= no

CLANG_ENZYME ?= $(HOME)/git/Enzyme/enzyme/buildRelease/Enzyme/ClangEnzyme-24.so
REACTANT_PATH ?= $(HOME)/git/Reactant/enzyme/buildRelease/Enzyme/libClangReactant-24.so
CLANG_PATH ?= $(HOME)/git/Enzyme/llvm-project/buildRelease/bin/clang++
LIB_RAISE_PATH ?= $(HOME)/git/Enzyme-JaX/bazel-bin/libRaise.so
REACTANT_EXTRA_PATH ?= $(HOME)/git/Reactant.jl/deps/ReactantExtra/bazel-bin/
BACKEND ?= cuda

export CUDA_HOME := $(CUDA_PATH)
export PATH := $(CUDA_PATH)/bin:$(PATH)
# The block size the kernels are launched with in simulation.cu, so the raised
# kernels are built for the same shape the hand-written ones run at.
BLOCKSIZE   ?= 32
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE = $(BLOCKSIZE)
export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)

ifneq ($(filter xla%,$(BACKEND)),)
  export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)
endif
OPTIMIZE    ?= yes

REMATERIALIZER ?= no
NEWCACHE    ?= yes
AA          ?= no
PHISTRUCT   ?= no
FORWARD     ?= yes
COALESE     ?= yes
CACHELICM   ?= yes
ABI         ?= no
ALLOCATOR   ?= no
MINCUT      ?= yes
READONLY    ?= yes
SPLITMULTI  ?= yes
AFFINEAD    ?= no
VERIFY      ?= no

DEBUG       = no
PROFILE     = no
SM_VERSION  = 120

#===============================================================================
# Program name & source code list
#===============================================================================

program = rsbench

source = \
main.cu \
simulation.cu\
io.cu \
init.cu \
material.cu \
utils.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -mllvm -max-heap-to-stack-size=1000000
# -mllvm -max-heap-to-stack-size=-1
# -Rpass=attributor -mllvm -debug -mllvm -debug-only=attributor

CC := $(CLANG_PATH)

ENZYME_LOAD_FLAGS :=
ifeq ($(USE_ENZYME_LLVM),yes)
  ENZYME_LOAD_FLAGS += -fplugin=$(CLANG_ENZYME)
else
  ifeq ($(EMBEDDED_CLANG),no)
    ENZYME_LOAD_FLAGS += -fplugin=$(REACTANT_PATH) -mllvm -raising-plugin-path -mllvm $(LIB_RAISE_PATH)
  endif
  CFLAGS += -mllvm -reactant-backend -mllvm $(BACKEND) -mllvm -reactant-dataflow

  ifeq ($(READONLY),yes)
    # Avoid caching loads of pointers that are never overwritten
    CFLAGS += -mllvm -reactant-mark-readonly
  endif

  ifeq ($(AFFINEAD),yes)
    CFLAGS += -mllvm -reactant-pre-ad-lower-affine=0
  else
    # Lower affine dialect operations before AD (https://github.com/EnzymeAD/Enzyme/issues/3031)
    CFLAGS += -mllvm -reactant-pre-ad-lower-affine=1
  endif

  ifeq ($(SPLITMULTI),yes)
    # Split certain multi-result ops into separate ops. Can lead to better mincut solutions
    CFLAGS += -mllvm -reactant-split-multi-results
  endif
endif

CFLAGS += -ffast-math --cuda-path=$(CUDA_PATH) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11 $(ENZYME_LOAD_FLAGS) -fno-exceptions

# Linker Flags
LDFLAGS = "-lcudart_static" "-ldl" "-lrt" -lpthread -lm

ifneq ($(filter xla%,$(BACKEND)),)
  LDFLAGS += -L$(REACTANT_EXTRA_PATH) -lReactantExtra
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS += -g -G
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
  CFLAGS += -DFORWARD=1
endif


ifeq ($(USE_ENZYME_LLVM),yes)

# Optimization Flags
ifeq ($(NEWCACHE),yes)
  CFLAGS += -mllvm -enzyme-new-cache=1
else
  CFLAGS += -mllvm -enzyme-new-cache=0
endif

ifeq ($(MINCUT),yes)
  CFLAGS += -mllvm -enzyme-mincut-cache=1
else
  CFLAGS += -mllvm -enzyme-mincut-cache=0
endif

# ifeq ($(AA),yes)
#   CFLAGS += -mllvm -enzyme-aggressive-aa=1
# else
#   CFLAGS += -mllvm -enzyme-aggressive-aa=0
# endif

ifeq ($(PHISTRUCT),yes)
  CFLAGS += -mllvm -enzyme-phi-restructure=1
else
  CFLAGS += -mllvm -enzyme-phi-restructure=0
endif

ifeq ($(INLINE),yes)
  CFLAGS += -DALWAYS_INLINE=1
endif

ifeq ($(CACHELICM),yes)
  CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=1
else
  CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=0
endif

ifeq ($(REMATERIALIZER),yes)
	CFLAGS +=  -mllvm -enzyme-rematerialize=1
else
	CFLAGS +=  -mllvm -enzyme-rematerialize=0
endif

endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) rsbench.cuh Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu rsbench.cuh Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) *.o *.re_export

# edit:
# 	vim -p $(source) rsbench.cuh

run:
	./$(program) -m event
