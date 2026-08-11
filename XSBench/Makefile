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
# The block size the kernels are launched with in Simulation.cu, so the raised
# kernels are built for the same shape the hand-written ones run at.
BLOCKSIZE   ?= 32
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE = $(BLOCKSIZE)
export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)

ifneq ($(filter xla%,$(BACKEND)),)
	export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)
endif
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

READONLY    ?= yes
SPLITMULTI  ?= yes
AFFINEAD    ?= no
RMATOMICS   ?= no
SORTBLOCK   ?= no
# Cap on registers per thread. The reverse kernel is latency-bound on scattered
# lookups, so occupancy can be worth more than the spills a cap introduces.
MAXREG      ?= no
# Promote the reverse pass's tape allocation out of the device heap. Without it
# the cache for the nuclide loop is a per-thread malloc/free on every lookup.
HEAP2STACK  ?= no
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
LDFLAGS = -lcudart_static "-ldl" "-lrt" -lpthread -lm

ifneq ($(filter xla%,$(BACKEND)),)
	LDFLAGS += -L$(REACTANT_EXTRA_PATH) -lReactantExtra
endif

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

	ifeq ($(RMATOMICS),yes)
		# Turn the atomics the reverse pass introduces back into load/store where
		# polyhedral analysis proves no race exists
		CFLAGS += -mllvm -reactant-remove-atomics=1
	else
		CFLAGS += -mllvm -reactant-remove-atomics=0
	endif

	ifeq ($(SORTBLOCK),yes)
		# Cluster the resulting loads at the top of their block so they overlap
		# instead of forming a chain of dependent round trips
		CFLAGS += -mllvm -reactant-sort-block-memory=1
	else
		CFLAGS += -mllvm -reactant-sort-block-memory=0
	endif
endif

CFLAGS += -ffast-math --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11 $(ENZYME_LOAD_FLAGS) -fno-exceptions

ifneq ($(MAXREG),no)
	CFLAGS += -Xcuda-ptxas -maxrregcount=$(MAXREG)
endif

ifneq ($(HEAP2STACK),no)
	CFLAGS += -mllvm -max-heap-to-stack-size=$(HEAP2STACK)
endif

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

$(program): $(obj) XSbench_header.cuh Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu XSbench_header.cuh Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.re_export

edit:
	vim -p $(source) XSbench_header.cuh

run:
	./$(program) -m event
