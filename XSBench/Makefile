#===============================================================================
# User Options
#===============================================================================
CUDA_PATH ?= /usr/local/cuda-11.7
CLANG_PATH ?= /home/wmoses/git/Polygeist/build/bin/cgeist -resource-dir /home/wmoses/git/Polygeist/mlir-build/lib/clang/16.0.0
# CLANG_PATH ?= /mnt/sabrent/wmoses/llvm13/buildall/bin/clang++

SM_VERSION  = 60


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
LDFLAGS = 

CC := $(CLANG_PATH)
CFLAGS := --function=* -struct-abi=1 --cpuify=distribute --cuda-lower --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11

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

ifeq ($(TEMPLATIZE),yes)
  CFLAGS += -DTEMPLATIZE=1
endif

ifeq ($(VERIFY),yes)
  CFLAGS += -DPRINT=1
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) XSbench_header.cuh Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu XSbench_header.cuh Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) XSbench_header.cuh

run:
	./$(program)
