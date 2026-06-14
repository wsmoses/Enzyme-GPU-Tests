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
export CUDA_HOME := $(CUDA_PATH)
export PATH := $(CUDA_PATH)/bin:$(PATH)
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE = 128
export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)

OPTIMIZE    ?= yes
FORWARD     ?= yes
RESTRICT    ?= yes
VERIFY      ?= no
DEBUG       ?= no
PROFILE     ?= no
SM_VERSION  ?= 120

#===============================================================================
# Program name & source code list
#===============================================================================

program = lulesh

source = allocator.cu lulesh-comms.cu lulesh-comms-gpu.cu lulesh.cu

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DNDEBUG -I. -mllvm -max-heap-to-stack-size=1000000

CC := $(CLANG_PATH)

ENZYME_LOAD_FLAGS :=
ifeq ($(EMBEDDED_CLANG),no)
  ENZYME_LOAD_FLAGS += -fplugin=$(ENZYME_PATH)
endif

CFLAGS += -ffast-math --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION) -std=c++11 $(ENZYME_LOAD_FLAGS) -Xclang -add-plugin -Xclang enzyme -mllvm -raising-plugin-path -mllvm $(LIB_RAISE_PATH) -mllvm -reactant-backend -mllvm cuda -x cuda -fno-exceptions

# Linker Flags
LDFLAGS = -lcudart_static "-ldl" "-lrt" -lpthread -lm -L$(REACTANT_EXTRA_PATH)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(RESTRICT),yes)
  CFLAGS += -DRESTRICT=1
else
  CFLAGS += -DRESTRICT=0
endif

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif

ifeq ($(FORWARD),yes)
  CFLAGS += -DNormal_forward=1
else
  CFLAGS += -DNormal_forward=0
endif

#===============================================================================
# Targets to Build
#===============================================================================

all: release

release: $(program)
debug: DEBUG=yes
debug: $(program)

$(program): $(source) Makefile
	$(CC) $(CFLAGS) $(source) $(LDFLAGS) -o $@

clean:
	rm -rf $(program) *.o *.re_export

run: $(program)
	./$(program)
