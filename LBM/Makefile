#===============================================================================
# User Options
#===============================================================================

CUDA_PATH ?= /usr/local/cuda-12.9

EMBEDDED_CLANG ?= no

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
FORWARD     ?= yes
COALESE     ?= yes
CACHELICM   ?= yes
ABI         ?= no
ALLOCATOR   ?= no
VERIFY      ?= no

DEBUG       = no
PROFILE     = no
SM_VERSION  = 120

#===============================================================================
# Program name & source code list
#===============================================================================

program = lbm

# lbm_kernel.cu
source = args.c lbm.cu parboil_cuda.c main.cc

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -mllvm -max-heap-to-stack-size=1000000 -I $(CUDA_PATH)/include -I .
# -mllvm -max-heap-to-stack-size=-1
# -Rpass=attributor -mllvm -debug -mllvm -debug-only=attributor


CC := $(CLANG_PATH)

ENZYME_LOAD_FLAGS :=
REACTANT_PLUGIN_FLAGS :=

ifeq ($(EMBEDDED_CLANG),no)
  ENZYME_LOAD_FLAGS += -fplugin=$(ENZYME_PATH)
  REACTANT_PLUGIN_FLAGS += -fplugin-arg-enzyme-raising-plugin-path=$(LIB_RAISE_PATH)
  REACTANT_PLUGIN_FLAGS += -fplugin-arg-enzyme-reactant-backend=cuda
else
  #reactant-clang
  REACTANT_PLUGIN_FLAGS += -Xclang -add-plugin -Xclang enzyme
  REACTANT_PLUGIN_FLAGS += -mllvm -reactant-backend -mllvm cuda
endif

CFLAGS += -ffast-math                       \
          --cuda-path=$(CUDA_PATH)          \
          -L$(CUDA_PATH)/lib64              \
          --cuda-gpu-arch=sm_$(SM_VERSION)  \
          -std=c++11                        \
          $(ENZYME_LOAD_FLAGS) $(REACTANT_PLUGIN_FLAGS) \
          -x cuda -fno-exceptions
  

# Linker Flags
LDFLAGS = -lcudart_static "-ldl" "-lrt" -lpthread -lm -L$(REACTANT_EXTRA_PATH) # -lReactantExtra

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
# lbm_kernel.cu
$(program): $(source) Makefile
	$(CC) $(CFLAGS) args.c lbm.cu parboil_cuda.c main.cc $(LDFLAGS) -o $@

clean:
	rm -f args.c.re_export lbm lbm.cu.re_export lbm_kernel.cu.re_export main.cc.re_export out.txt parboil_cuda.c.re_export 

run: $(program)
	./$(program) 10 -o out.txt
