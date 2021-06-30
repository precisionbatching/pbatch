# PBatch

# Install PrecisionBatching & Cutlass Kernels
Add to pythonpaths:
```
source setup.sh
```

PrecisionBatching:
```
pushd src/pytorch/cpp/pbatch/; bash install.sh; popd
```

Cutlass:
```
pushd src/pytorch/cpp/cutlass_linear/; bash install.sh; popd
```

# Main PrecisionBatching Kernel Files

```
src/pytorch/cpp/pbatch/pbatch.h 
```

Main entry point: `pbatch_kernel` which calls one of `pbatch_inner_tiled_act32`, `pbatch_inner_tiled_act16`, or `pbatch_inner_tiled_act8` (depending on activation precision, which is compiled into code for optimization reasons; one may use any level of activation precision by changing the code, for example, adding a `pbatch_inner_tiled_act24` method).

Note that, `pbatch_cvt_float_fixed_trans` is the bitwise matrix transpose GPU kernel, which converts floating point input matrix to fixed point matrix, views the fixed point matrix as a set of bits, then transposes this bit matrix.