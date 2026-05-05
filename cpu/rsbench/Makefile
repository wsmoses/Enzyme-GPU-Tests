LLVM_BUILD = $(HOME)/git/llvm-project/build
CC=$(LLVM_BUILD)/bin/clang++
CLANG_ENZYME ?= $(HOME)/git/Enzyme/build/Enzyme/ClangEnzyme-23.so
LLVM_ENZYME ?= $(HOME)/git/Enzyme/build/Enzyme/LLVMEnzyme-23.so
ENZYME_OPT ?= $(HOME)/.local/bin/enzymexlamlir-opt
OPT ?= $(LLVM_BUILD)/bin/opt
MLIR_TRANSLATE ?= $(LLVM_BUILD)/bin/mlir-translate

# No enzyme inlining?
PASS_PIPELINE ?= inline{default-pipeline=canonicalize max-iterations=4},sroa-wrappers{set_private=false attributor=false},gpu-launch-recognition,canonicalize,libdevice-funcs-raise,canonicalize,symbol-dce,parallel-lower{wrapParallelOps=true},llvm-to-memref-access,polygeist-mem2reg,canonicalize,convert-llvm-to-cf,canonicalize,polygeist-mem2reg,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),llvm.func(canonicalize-loops),canonicalize-scf-for,canonicalize,inline{max-iterations=4},affine-cfg,canonicalize,func.func(canonicalize-loops),llvm.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,llvm-to-affine-access,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,print{filename=rsbench-raised.mlir},lower-affine,enzyme,canonicalize,remove-unnecessary-enzyme-ops,flatten-enzyme-caches,enzyme-simplify-math,symbol-dce,lower-affine,convert-parallel-to-gpu1,gpu-kernel-outlining,canonicalize,convert-parallel-to-gpu2,lower-affine,parallel-serialization,canonicalize,print{filename=rsbench-before-llvm.mlir},convert-polygeist-to-llvm{backend=cpu},strip-gpu-info,gpu-module-to-binary

DIFF_PASSES ?= lower-affine,enzyme,canonicalize,remove-unnecessary-enzyme-ops,flatten-enzyme-caches,enzyme-simplify-math
LOWER_PASS_PIPELINE ?= symbol-dce,lower-affine,convert-parallel-to-gpu1,gpu-kernel-outlining,canonicalize,convert-parallel-to-gpu2,lower-affine,parallel-serialization,canonicalize,print{filename=rsbench-before-llvm.mlir},convert-polygeist-to-llvm{backend=cpu},strip-gpu-info,gpu-module-to-binary

source = material.cpp init.cpp test_driver.cpp

obj = tmp/material.o tmp/init.o tmp/test_driver.o

ENZYME_FLAGS = -Xclang -load -Xclang $(CLANG_ENZYME)
CFLAGS= -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -ffast-math -O2
LDFLAGS = -lm

rsbench: $(obj) tmp/simulation-ref.o rsbench.h Makefile
	$(CC) $(CFLAGS) $(obj) tmp/simulation-ref.o -o $@ $(LDFLAGS)

rsbench_mlir: $(obj) tmp/simulation_mlir.o rsbench.h Makefile
	$(CC) $(CFLAGS) $(obj) tmp/simulation_mlir.o -o $@ $(LDFLAGS)

tmp/simulation.ll: simulation.cpp rsbench.h Makefile
	$(CC) $(CFLAGS) simulation.cpp -S -emit-llvm -o $@

tmp/simulation-ref.o: tmp/simulation-ref.ll
	$(CC) $(CFLAGS) $< -c -o $@

tmp/simulation-ref.ll: tmp/simulation.ll
	$(OPT) -S -load-pass-plugin=$(LLVM_ENZYME) $< -passes="enzyme,mem2reg" -o $@

tmp/simulation-start.mlir: tmp/simulation.ll
	$(MLIR_TRANSLATE) $< -import-llvm -o $@

# tmp/simulation-diff.mlir: rsbench-diff.mlir
# 	$(ENZYME_OPT) $< --pass-pipeline="builtin.module($(LOWER_PASS_PIPELINE))" -o $@

tmp/simulation-diff.mlir: tmp/simulation-start.mlir
	$(ENZYME_OPT) $< --pass-pipeline="builtin.module($(PASS_PIPELINE))" -o $@

tmp/simulation-diff.ll: tmp/simulation-diff.mlir
	$(MLIR_TRANSLATE) $< -mlir-to-llvmir -o $@

tmp/simulation_mlir.o: tmp/simulation-diff.ll
	$(CC) -c $< -o $@

tmp/%.o: %.cpp rsbench.h Makefile
	$(CC) $(CFLAGS) $(ENZYME_FLAGS) -c $< -o $@

clean:
	rm -rf *.o
