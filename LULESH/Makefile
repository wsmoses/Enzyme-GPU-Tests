NVCC		= /home/lpaehler/Work/temporary_files/llvm-project/build/bin/clang++
# FLAGS		= -arch=sm_86
# DFLAGS	= -lineinfo

# Ablation testing & numerical verification
VERIFY  ?= no
FORWARD ?= yes
AGGRPHIOPT ?= no
PHIOPT ?= no
BRANCHYOPT ?= no
MINCCACHE ?= yes
NEWCACHE ?= yes

RFLAGS 	= -DNDEBUG  
CFLAGS  = -O3 -DNDEBUG -g -fno-exceptions -mllvm -enzyme-print -Rpass=enzyme -fno-experimental-new-pass-manager\
 	-std=c++17 --cuda-path=/usr/local/cuda-11.2 -L/usr/local/cuda-11.2/lib64 --cuda-gpu-arch=sm_80\
	--no-cuda-version-check -Xclang -load -Xclang /home/lpaehler/Work/temporary_files/Enzyme/build/Enzyme/ClangEnzyme-13.so
LDFLAGS = -lcudart_static -ldl -lrt -lpthread -lm 


ifeq ($(VERIFY),yes)
	CFLAGS += -DVERIFY
endif

ifeq ($(MINCCACHE),yes)
	CFLAGS += -mllvm -enzyme-mincut-cache=1
else
	CFLAGS += -mllvm -enzyme-mincut-cache=0
endif

ifeq ($(PHIOPT),no)
	CFLAGS += -mllvm -enzyme-disable-phi=1
endif

ifeq ($(NEWCACHE),yes)
	CFLAGS += -mllvm -enzyme-new-cache=1
else
	CFLAGS += -mllvm -enzyme-new-cache=0
endif

ifeq ($(BRANCHYOPT),yes)
	CFLAGS += -DBRANCHYOPT
endif

ifeq ($(AGGRPHIOPT),yes)
	CFLAGS += -mllvm -enzyme-phi-restructure=1
else
	CFLAGS += -mllvm -enzyme-phi-restructure=0
endif

ifeq ($(FORWARD),yes)
	CFLAGS += -DNormal_forward=1
else
	CFLAGS += -DNormal_forward=0
endif
#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

#LINKFLAGS = -lmpich -L$(MPICH_DIR)/lib 
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

all: release 

debug: LINKFLAGS += 

release: 	FLAGS += $(RFLAGS)
debug: 		FLAGS += $(DFLAGS)

release: lulesh
debug: lulesh

lulesh: allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o
	$(NVCC) $(CFLAGS) allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o -o lulesh $(LDFLAGS)

allocator.o: allocator.cu vector.h
	$(NVCC) $(CFLAGS) allocator.cu -I ./ -c -o allocator.o

lulesh.o: lulesh.cu util.h vector.h allocator.h
	$(NVCC) $(CFLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o lulesh.o

lulesh-comms.o: lulesh-comms.cu
	$(NVCC) $(CFLAGS) lulesh-comms.cu -I ./ -c -o lulesh-comms.o

lulesh-comms-gpu.o: lulesh-comms-gpu.cu
	$(NVCC) $(CFLAGS) lulesh-comms-gpu.cu -I ./ -c -o lulesh-comms-gpu.o

clean: 
	rm -rf allocator.o lulesh-comms.o lulesh-comms-gpu.o lulesh.o lulesh xyz.asc regNumList.txt
