# Enzyme-GPU-Tests
[![DOI](https://zenodo.org/badge/369262862.svg)](https://zenodo.org/badge/latestdoi/369262862)

This repo contains the benchmarks for Enzyme on GPU's.

If Enzyme, or part of this repository is useful to you, please cite:
```
@inproceedings{enzymeGPU,
  title={Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme},
  author={Moses, William S and Churavy, Valentin and Paehler, Ludger and H{\"u}ckelheim, Jan and Hari Krishna Narayanan, Sri and Schanen, Michel and Doerfert, Johannes},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  year={2021},
  location = {St. Louis, Missouri},
  series = {SC '21}
}
```

Below we describe how to run the 6 benchmarks presented here: XSBench (CUDA), RSBench (CUDA), Parboil LBM (CUDA), LULESH (CUDA), DG (CUDA), DG (AMD). Of these benchmarks, the AMD and CUDA DG codes are Julia-based where as the remainder are C++/CUDA. Evaluation of these benchmarks allowed the paper to demonstrate the efficiency of the generated gradients in comparison to the original code, the impact of the novel optimizations, the scalability of the generated gradients, and the correctness of the tool.

To run the benchmarks used in the paper, we first need to build the LLVM compiler toolchain before we can subsequently link the compiler plugin of Enzyme against our built LLVM version. To install LLVM, please follow the following steps:

```
$ cd ~
$ git clone https://github.com/llvm/llvm-project
$ cd llvm-project
# The following git hash was used in our paper
# You may use this, or a newer version
$ git checkout 8dab25954b0acb53731c4aa73e9a7f4f98263030
$ mkdir build && cd build
$ cmake ../llvm -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DCMAKE_BUILD_TYPE=Release -G Ninja
$ ninja
# This may take a while
# clang is now be available in ~/llvm-project/build/bin/clang
```
We now must build an Enzyme based off of our chosen LLVM version.  

```
$ cd ~
$ git clone https://github.com/wsmoses/Enzyme
# The following git hash was used in our paper
# You may use this, or a newer version
$ git checkout ec75831a8cb0170090c366f8da6e3b2b87a20f6e
$ cd Enzyme/enzyme
$ mkdir build && cd build
$ cmake ../enzyme -DLLVM_DIR=/path/to/llvm/build -DCMAKE_BUILD_TYPE=Release -G Ninja
$ ninja
# ClangEnzyme-13.so will now be available in ~/Enzyme/enzyme/build/Enzyme/ClangEnzyme-13.so
```

Some of the C++ benchmarks require a custom CUDA libdevice (the implementation of various CUDA intrinsics). This is to remedy an issue within LLVM that prevents common math functions from being identified as LLVM intrinsics (this is being worked on in upstream LLVM). For the default CUDA installation, libdevice can be found at `/usr/local/cuda/nvvm/libdevice.10.bc`. The following code snippet describes how to replace a libdevice file, assuming you are using CUDA 11.2. The instructions are similar for a different CUDA installation, with the path changed accordingly. Note that you may need to be root to perform the change, and that you should always make a backup of your previous libdevice.

```
# Save a copy of your current libdevice
$ sudo cp /usr/local/cuda-11.2/nvvm/libdevice/libdevice.10.bc /usr/local/cuda-11.2/nvvm/libdevice/libdevice.10.bc.old
$ sudo cp /path/to/new/libdevice.10.bc /usr/local/cuda-11.2/nvvm/libdevice/libdevice.10.bc
```

We have created Python3 scripts to ease setting up and running our experiments. They will attempt to deduce appropriate paths given the following environmental variables. In the event that there is an issue, you likely may need to change `bench.py` or the `Makefile` for the test as appropriate. All of the `bench.py` benchmarking scripts follow the same structure.

```
# The index of the CUDA GPU desired
$ export DEVICE=1
# The path to the CUDA installation
$ export CUDA_PATH=/usr/local/cuda-11.2
# The path to the Clang++ binary we built above
$ export CLANG_PATH=/path/to/llvm/build/bin/clang++
# The path to the Enzyme plugin we built above
$ export ENZYME_PATH=/path/to/Enzyme/enzyme/build/Enzyme/ClangEnzyme-13.so
```

We can now work with the benchmark suite (this repository).

```
$ git clone https://github.com/wsmoses/Enzyme-GPU-Tests
```

The benchmark suite folder breaks down into the following structure:
* DG, a Discontinuous Galerkin benchmark (CUDA & ROCm in Julia)
* LBM, a Lattice Boltzmann benchmark (CUDA in C++) 
* LULESH, a Lagrangian Hydrodynamics benchmark (CUDA in C++)
* RSBench, a Monte Carlo Particle Transport benchmark (CUDA in C++)
* XSBench, a Monte Carlo Particle Transport benchmark (CUDA in C++)

We can now enter one of the 4 C++ test directories (XSBench, RSBench, LBM, LULESH) and run the corresponding benchmark.

```
$ cd Enzyme-GPU-Tests/LBM
$ python3 bench.py
# output of benchmark times printed out here
```

The `bench.py` script will run first an ablation analysis that enables or disables differentiation, along with several optimizations. The result of these tests will be the execution time of the gradient and/or original kernel. The script will then run by scaling tests for both the gradient and original kernel by evaluating on increasing problem sizes. Some benchmarks (XSBench, LBM, LULESH, DG (CUDA)) will end by printing out the derivative as computed by both numeric differentiation and Enzyme. These will include `VERIFY=yes` as part of the run line. All other run lines will contain the execution time of that benchmark. Be aware that LULESH's ablation analysis includes benchmarks configurations, which do not perform compiler optimizations and are hence significantly slower than the other benchmarks.

XSBench and RSBench require the libdevice found in Enzyme-GPU-Tests/libdevice1.

LBM uses the packaged libdevice from NVIDIA.

LULESH uses the libdevice found in Enzyme-GPU-Tests/libdevice2. The LULESH benchmark suite furthermore relies on NVIDIA's N-Sight compute utility (NCU). NCU has known issues with access to the GPU Performance counters, which you will need to benchmark the gradient kernel of LULESH. If you should run into this issue please have a look at the [following documentation](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nvprof) to remedy it.

Odd performance results or a compiler error is a potential indicator of using an incorrect libdevice.

For example, when compiling one of the ablation tests of RSBench without the correct libdevice, one may see the following error when running `bench.py`: 
```
cannot handle (augmented) unknown intrinsic %5 = tail call i32 @llvm.nvvm.d2i.hi(double %0)
#21 fatal error: error in backend: (augmented) unknown intrinsic
clang-13: error
```

The two DG tests were run using Julia 1.6. Julia at this version must be found in your path before being able to run the Julia tests. To obtain a working Julia installation see [here](https://julialang.org/downloads/) and follow the provided installation instructions.

DG (CUDA) was run with the libdevice found in Enzyme-GPU-Tests/libdevice1.

We have provided a similar `bench.py` script for DG. While printed in a different format (CSV-style), it contains the same information about runtimes for both ablation and scaling as the C++ CUDA tests (DG AMD does not have an ablation analysis as it does not run without all optimizations applied).

```
$ cd Enzyme-GPU-Tests/DG/cuda
$ python3 bench.py
```

Note that the numeric verification may come earlier in the script's output, and should look something like this: 
```
# Enzyme derivative as the first element of tuple
# followed by numeric approximation on the right
(dQ.dval[1], (o2 - o1) / 0.0001) = (-1.105959f0, [-1.10626220703125])
```

The forward pass alone in the CSV-style output are denoted as the "primal" rows, whereas the derivative runtimes are marked as "all_dub".

The DG tests may require additional setup. For example, you see an output like below (note that this may also occur if you try to run DG (AMD) on a system without the relevant AMD libraries available). 
```
Warning: HSA runtime has not been built, runtime functionality will be unavailable. Please run Pkg.build("AMDGPU") and reload AMDGPU.
```

You may then need to explicitly run various setup routines within Julia's package manager. To fix the Julia setup for the test, perform the following to enter an interactive shell.

```
$ cd Enzyme-GPU-Tests/DG/rocm
$ julia --project=. julia> using Pkg; Pkg.build("AMDGPU")
```
