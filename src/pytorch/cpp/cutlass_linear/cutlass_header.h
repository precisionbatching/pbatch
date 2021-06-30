#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "cutlass/gemm/gemm.h"

#include "tools/test/perf/gemm/gemm_profiler.h"
#include "tools/test/perf/gemm/gemm_perf_testbed.h"

#include "cutlass/wmma_matrix.h"

#ifdef CUTLASS_USE_WMMA_API
#ifdef CUTLASS_USE_SUBBYTE_WMMA
#pragma warning( disable : 4503)

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#include "tools/test/perf/gemm/cutlass_dispatch.h"
#include "cutlass/gemm/sgemm_traits.h"

#endif
#endif

#ifndef CUTLASS_HEADER
#define CUTLASS_HEADER
cublasHandle_t cutlass_handle;

static int cutlass_initialized = 0;

void cutlass_initialize() {
  if (cutlass_initialized) return;
  cutlass_initialized = 1;
  
  // Cublas handle
  cublasCreate(&cutlass_handle);
}


#endif
