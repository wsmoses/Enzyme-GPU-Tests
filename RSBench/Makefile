#===============================================================================
# User Options
#===============================================================================
CUDA_PATH ?= /usr/local/cuda-11.2

ENZYME_PATH ?= /home/wmoses/git/Enzyme/enzyme/build13Fast/Enzyme/ClangEnzyme-13.so
CLANG_PATH ?= /mnt/sabrent/wmoses/llvm13/buildallfast/bin/clang++

# ENZYME_PATH ?= /home/wmoses/git/Enzyme/enzyme/build13D/Enzyme/ClangEnzyme-13.so
# CLANG_PATH ?= /mnt/sabrent/wmoses/llvm13/buildall/bin/clang++

OPTIMIZE    ?= yes

NEWCACHE    ?= yes
MINCUT      ?= no

AA          ?= no
PHISTRUCT   ?= yes
FORWARD     ?= no
INLINE      ?= yes
CACHELICM   ?= yes

DEBUG       = no
PROFILE     = no
SM_VERSION  = 60

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
CFLAGS := -fno-experimental-new-pass-manager --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11 -Xclang -load -Xclang $(ENZYME_PATH) 

# Linker Flags
LDFLAGS = "-lcudart_static" "-ldl" "-lrt" -lpthread -lm

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

ifeq ($(AA),yes)
  CFLAGS += -mllvm -enzyme-aggressive-aa=1
else
  CFLAGS += -mllvm -enzyme-aggressive-aa=0
endif


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



#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) rsbench.cuh Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu rsbench.cuh Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf rsbench $(obj)

edit:
	vim -p $(source) rsbench.cuh

run:
	./rsbench
