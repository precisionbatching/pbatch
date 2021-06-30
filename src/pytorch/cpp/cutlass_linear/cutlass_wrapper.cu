// Wrappers around cutlass cuda code, to be called by pytorch interface

#ifndef CUTLASS_WRAPPERS
#define CUTLASS_WRAPPERS

#include "cutlass_header.h"
#include "cutlass_fp32.h"
#include "cutlass_fp16.h"
#include "cutlass_int8.h"
#include "cutlass_int4.h"
#include "cutlass_int1.h"

void cutlass_linear_fp32_kernel_wrapper(float *input,
					float *weight,
					float *bias,
					float *output,
					int M, int N, int K) {
  cutlass_linear_fp32_kernel(input, 
			     weight,
			     bias,
			     output,
			     M, N, K);
}


void cutlass_linear_fp16_kernel_wrapper(float *input,
					void *input_int,
					void *weight,
					float *bias,
					void *output_int,
					float *output,
					int M, int N, int K) {
  cutlass_linear_fp16_kernel(input,
			     (half *)input_int,
			     (half *)weight,
			     bias,
			     (half *)output_int,
			     output,
			     M, N, K);
}


void cutlass_linear_int8_kernel_wrapper(float *input,
					signed char *input_int,
					signed char *weight,
					float scale,
					float *bias,
					int *output_int,
					float *output,
					int M, int N, int K) {
  cutlass_linear_int8_kernel(input, input_int,
			     weight, scale, bias,
			     output_int, output,
			     M, N, K);
}

void cutlass_linear_int4_kernel_wrapper(float *input,
					void *input_int,
					void *weight,
					float scale,
					float *bias,
					int *output_int,
					float *output,
					int M, int N, int K) {
  cutlass_linear_int4_kernel(input, input_int,
			     weight, scale, bias,
			     output_int, output,
			     M, N, K);
}

void cutlass_linear_int1_kernel_wrapper(float *input,
					void *input_int,
					void *weight,
					float scale,
					float *bias,
					int *output_int,
					float *output,
					int M, int N, int K) {
  cutlass_linear_int1_kernel(input, input_int,
			     weight, scale, bias,
			     output_int, output,
			     M, N, K);
}

#endif
