#===============================================================================
# User Options
#===============================================================================

CUDA_PATH ?= /usr/local/cuda-12.9

ENZYME_PATH ?= $(HOME)/git/Reactant/enzyme/buildRelease/Enzyme/ClangEnzyme-22.so
CLANG_PATH ?= $(HOME)/git/Enzyme/llvm-project/buildRelease/bin/clang++
LIB_RAISE_PATH ?= $(HOME)/git/Enzyme-JaX/bazel-bin/libRaise.so
REACTANT_EXTRA_PATH ?= $(HOME)/git/Reactant.jl/deps/ReactantExtra/bazel-bin/

export REACTANT_DEBUG=1
export PATH := $(CUDA_PATH)/bin:$(PATH)
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE = 128
export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)

OPTIMIZE    ?= no # yes

NEWCACHE    ?= yes
AA          ?= no
PHISTRUCT   ?= no
FORWARD     ?= no
COALESE     ?= yes
CACHELICM   ?= yes
ABI         ?= no
ALLOCATOR   ?= no
VERIFY      ?= no

DEBUG       = yes
PROFILE     = no
SM_VERSION  = 120

#===============================================================================
# Program name & source code list
#===============================================================================

program = lbm

source = args.c lbm.cu lbm_kernel.cu parboil_cuda.c main.cc

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -mllvm -max-heap-to-stack-size=1000000 -I $(CUDA_PATH)/include -I .
# -mllvm -max-heap-to-stack-size=-1
# -Rpass=attributor -mllvm -debug -mllvm -debug-only=attributor


CC := $(CLANG_PATH)
CFLAGS += -ffast-math --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11 -Xclang -load -Xclang $(ENZYME_PATH) -Xclang -add-plugin -Xclang enzyme -mllvm -raising-plugin-path -mllvm $(LIB_RAISE_PATH) -mllvm -reactant-backend -mllvm cuda -x cuda -fno-exceptions
  

# Linker Flags
# LDFLAGS = -lcudart_static "-ldl" "-lrt" -lpthread -lm -L$(REACTANT_EXTRA_PATH) -lReactantExtra
LDFLAGS = -lcudart_static "-ldl" "-lrt" -lpthread -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(ALLOCATOR),yes)
  CFLAGS += -DALLOCATOR
endif

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif

ifeq ($(ABI),yes)
  CFLAGS += -DABI
else
  CFLAGS += -DSIZE=8
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
  CFLAGS += 
else
  CFLAGS += -DALLOW_AD=1 
endif


# Optimization Flags
# ifeq ($(NEWCACHE),yes)
#   CFLAGS += -mllvm -enzyme-new-cache=1 -mllvm -enzyme-mincut-cache=1
#   ifeq ($(ABI),yes)
#     CFLAGS += -DSIZE=20
#   endif
# else
#   CFLAGS += -mllvm -enzyme-new-cache=0 -mllvm -enzyme-mincut-cache=0
#   ifeq ($(ABI),yes)
#     CFLAGS += -DSIZE=80
#   endif
# endif

# ifeq ($(AA),yes)
#   CFLAGS += -mllvm -enzyme-aggressive-aa=1
# else
#   CFLAGS += -mllvm -enzyme-aggressive-aa=0
# endif


# ifeq ($(PHISTRUCT),yes)
#   CFLAGS += -mllvm -enzyme-phi-restructure=1
# else
#   CFLAGS += -mllvm -enzyme-phi-restructure=0
# endif

# ifeq ($(COALESE),yes)
#   CFLAGS += -mllvm -enzyme-coalese
# endif

# ifeq ($(CACHELICM),yes)
#   CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=1
# else
#   CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=0
# endif



#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(source) Makefile
	$(CC) $(CFLAGS) args.c lbm.cu lbm_kernel.cu parboil_cuda.c main.cc $(LDFLAGS) -o $@

clean:
	rm -rf $(program) *.o *.re_export

run: $(program)
	./$(program) 10 -o out.txt
