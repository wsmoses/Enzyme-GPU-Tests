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

export REACTANT_DEBUG=1
export PATH := $(CUDA_PATH)/bin:$(PATH)
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE = 128
export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)

OPTIMIZE    ?= yes

NEWCACHE    ?= yes
AA          ?= no
PHISTRUCT   ?= no
FORWARD     ?= yes
COALESE     ?= yes
CACHELICM   ?= yes
ABI         ?= yes
ALLOCATOR   ?= no
READONLY    ?= yes
SPLITMULTI  ?= yes
AFFINEAD    ?= yes
VERIFY      ?= no

DEBUG       = no
PROFILE     = no
SM_VERSION  = 120

#===============================================================================
# Program name & source code list
#===============================================================================

program = lbm

source = args.cc lbm.cu parboil_cuda.cc main.cc

obj = $(patsubst %.cc,%.o,$(patsubst %.cu,%.o,$(source)))

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -mllvm -max-heap-to-stack-size=1000000 -I $(CUDA_PATH)/include -I .
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
  CFLAGS += -mllvm -reactant-backend -mllvm cuda -mllvm -reactant-dataflow

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

CFLAGS += -ffast-math --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11 $(ENZYME_LOAD_FLAGS) -fno-exceptions
  

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
  CFLAGS += -DSIZE=20
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
  ifeq ($(USE_ENZYME_LLVM),yes)
    CFLAGS += -DENZYME_LLVM_AD=1
  endif
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

$(program): $(obj) lbm_kernel.cu Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu lbm.h main.h parboil.h lbm_macros.h lbm_kernel.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cc lbm.h main.h parboil.h lbm_macros.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) *.o *.re_export

run: $(program)
	./$(program) 10 -o out.txt
