# PrecisionBatching

# Overview
This repository contains the source code for PrecisionBatching: Bitserial Decomposition for Efficient Neural Network Inference on GPUs.

# Hardware Requirements
For general reproducibility, run on a T4 gpu (or a GPU with architecture cc7.5).

# Install Conda Environment
```
conda create -n pbatch --file conda_req.txt
```
```
pip install -r pip_req.txt
```

# Install PrecisionBatching & Cutlass Kernels

Activate conda environment:
```
conda activate pbatch
```

Add code to pythonpaths:
```
source setup.sh
```

Install pbatch and cutlass kernels:
```
bash install.sh
```

# Running

## Kernel Benchmarks

Outputs the kernel benchmark speedups (Table 1 in the paper).
```
bash scripts/kernel_performance.sh
```

## End to End Inference Speedups

``bash scripts/mnist_inference.sh```
``bash scripts/lm_inference.sh```
``bash scripts/snli_inference.sh```

These output [mnist|lm|snli].pdf which shows the end to end speedups of pbatch vs standard quantized inference (Figure 4 in the paper).
Note to avoid long running times, we evaluate architectures where each layer has the same weight and activation precision (hence only PBatch/Cutlass Uniform is outputted).

Note, we do not provide scripts for the RL tasks, as running these tasks required a paid license (Mujoco); however, our source code for these tasks are in the repository.

# Main PrecisionBatching Kernel Files

```
src/pytorch/cpp/pbatch/pbatch.h 
```

Main entry point: `pbatch_kernel` which calls one of `pbatch_inner_tiled_act32`, `pbatch_inner_tiled_act16`, or `pbatch_inner_tiled_act8` (depending on activation precision, which is compiled into code for optimization reasons; one may use any level of activation precision by changing the code, for example, adding a `pbatch_inner_tiled_act24` method).

Note that, `pbatch_cvt_float_fixed_trans` is the bitwise matrix transpose GPU kernel, which converts floating point input matrix to fixed point matrix, views the fixed point matrix as a set of bits, then transposes this bit matrix.