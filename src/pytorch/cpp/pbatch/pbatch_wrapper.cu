// Wrappers around pbatch cuda code, to be called by pytorch interface

#ifndef PBATCH_WRAPPERS
#define PBATCH_WRAPPERS

#include "pbatch.h"
#include <iostream>
#include <vector>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

void pbatch_kernel_wrapper(void *w,
			   float *input,
			   float *bias,
			   void *output,
			   int act_bits,
			   float act_shift_scale) {
  PBatchWeightLayers *weight_layers = (PBatchWeightLayers *)w;
  PBatchOutput *C = (PBatchOutput *)output;
  
  int M = weight_layers->M;
  int K = weight_layers->K;
  int N = 1;

  pbatch_kernel(weight_layers,
		input,
		bias,
		C,
		M, K, N,
		act_bits,
		act_shift_scale);
}


torch::Tensor &pbatch_fetch_result_wrapper(void *output) {
  return ((PBatchOutput *)output)->result;
}

#endif
