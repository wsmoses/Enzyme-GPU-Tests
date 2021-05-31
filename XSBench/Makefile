#===============================================================================
# User Options
#===============================================================================
CUDA_PATH ?= /usr/local/cuda-11.2

ENZYME_PATH ?= /home/wmoses/git/Enzyme/enzyme/build13Fast/Enzyme/ClangEnzyme-13.so
CLANG_PATH ?= /mnt/sabrent/wmoses/llvm13/buildallfast/bin/clang++

# ENZYME_PATH ?= /home/wmoses/git/Enzyme/enzyme/build13D/Enzyme/ClangEnzyme-13.so
# CLANG_PATH ?= /mnt/sabrent/wmoses/llvm13/buildall/bin/clang++

SM_VERSION  = 60


OPTIMIZE    ?= yes
FORWARD     ?= no
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
LDFLAGS = "-lcudart_static" "-ldl" "-lrt" -lpthread -lm

CC := $(CLANG_PATH)
CFLAGS := -fno-experimental-new-pass-manager --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11 -Xclang -load -Xclang $(ENZYME_PATH) 

CFLAGS += -mllvm -simplifycfg-sink-common=0

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
ifeq ($(NEWCACHE),yes)
  CFLAGS += -mllvm -enzyme-new-cache=1
else
  CFLAGS += -mllvm -enzyme-new-cache=0
endif


ifeq ($(DYN),yes)
  CFLAGS += -mllvm -enzyme-inactive-dynamic=1
else
  CFLAGS += -mllvm -enzyme-inactive-dynamic=0
endif

ifeq ($(MINCACHE),yes)
  CFLAGS += -mllvm -enzyme-mincut-cache=1
else
  CFLAGS += -mllvm -enzyme-mincut-cache=0
endif

ifeq ($(SELECT),yes)
  CFLAGS += -mllvm -enzyme-select-opt=1
else
  CFLAGS += -mllvm -enzyme-select-opt=0
endif

ifeq ($(SPECPHI),yes)
  CFLAGS += -mllvm -enzyme-speculate-phis=1
else
  CFLAGS += -mllvm -enzyme-speculate-phis=0
endif

ifeq ($(CACHELICM),yes)
  CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=1
else
  CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=0
endif

ifeq ($(AA),yes)
  CFLAGS += -mllvm -enzyme-aggressive-aa=1
else
  CFLAGS += -mllvm -enzyme-aggressive-aa=0
endif

ifeq ($(COALESE),yes)
  CFLAGS += -mllvm -enzyme-coalese
endif

ifeq ($(PHISTRUCT),yes)
  CFLAGS += -mllvm -enzyme-phi-restructure=1
else
  CFLAGS += -mllvm -enzyme-phi-restructure=0
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
