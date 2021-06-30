// Functions called in pytorch code.
  
#include <torch/extension.h>
#include <stdio.h>
#include <vector>

/***********************
 * External Functions  *
 ***********************/
void cutlass_initialize();

void cutlass_linear_fp16_kernel_wrapper(float *input,
					void *input_int,
					void *weight,
					float *bias,
					void *output_half,
					float *output,
					int M, int N, int K);

void cutlass_linear_fp32_kernel_wrapper(float *input,
					float *weight,
					float *bias,
					float *output,
					int M, int N, int K);

void cutlass_linear_int8_kernel_wrapper(float *input,
					signed char *input_int,
					signed char *weight,
					float scale,
					float *bias,
					int *output_int,
					float *output,
					int M, int N, int K);

void cutlass_linear_int4_kernel_wrapper(float *input,
					void *input_int,
					void *weight,
					float scale,
					float *bias,
					int *output_int,
					float *output,
					int M, int N, int K);

void cutlass_linear_int1_kernel_wrapper(float *input,
					void *input_int,
					void *weight,
					float scale,
					float *bias,
					int *output_int,
					float *output,
					int M, int N, int K);

/******************************
 * Cutlass Pytorch Interfaces *
 ******************************/

void cutlass_linear_fp32(torch::Tensor &input,
			 torch::Tensor &weight,
			 torch::Tensor &bias,
			 torch::Tensor &output) {
  
  // Resize output if necessary
  output.resize_({weight.size(0), input.size(0)});
  
  // Call cuda kernel
  int M = weight.size(0);
  int K = weight.size(1);
  int N = input.size(0);

  cutlass_linear_fp32_kernel_wrapper(input.data<float>(),
				     weight.data<float>(),
				     bias.data<float>(),
				     output.data<float>(),
				     M, N, K);
}

void cutlass_linear_fp16(torch::Tensor &input,
			 torch::Tensor &input_int,
			 torch::Tensor &weight,
			 torch::Tensor &bias,
			 torch::Tensor &output_int,
			 torch::Tensor &output) {
  
  // Resize output if necessary
  output.resize_({weight.size(0), input.size(0)});
  output_int.resize_({weight.size(0), input.size(0)});
  input_int.resize_({input.size(0), input.size(1)});
  
  // Call cuda kernel
  int M = weight.size(0);
  int K = weight.size(1);
  int N = input.size(0);

  cutlass_linear_fp16_kernel_wrapper(input.data<float>(),
				     input_int.data_ptr(),
				     weight.data_ptr(), 
				     bias.data<float>(),
				     output_int.data_ptr(),
				     output.data<float>(),
				     M, N, K);
}


void cutlass_linear_int8(torch::Tensor &input,
			 torch::Tensor &input_int,
			 torch::Tensor &weight,
			 float scale,
			 torch::Tensor &bias,
			 torch::Tensor &output_int,
			 torch::Tensor &output) {
  
  // Resize output if necessary
  output.resize_({weight.size(0), input.size(0)});
  output_int.resize_({weight.size(0), input.size(0)});
  input_int.resize_({input.size(0), input.size(1)});
  
  // Call cuda kernel
  int M = weight.size(0);
  int K = weight.size(1);
  int N = input.size(0);
  
  cutlass_linear_int8_kernel_wrapper(input.data<float>(),
				     input_int.data<signed char>(),
				     weight.data<signed char>(),
				     scale,
				     bias.data<float>(),
				     output_int.data<int>(),
				     output.data<float>(),
				     M, N, K);
}

void cutlass_linear_int4(torch::Tensor &input,
			 torch::Tensor &input_int,
			 torch::Tensor &weight,
			 float scale,
			 torch::Tensor &bias,
			 torch::Tensor &output_int,
			 torch::Tensor &output) {
  
  // Resize output if necessary
  output.resize_({weight.size(0), input.size(0)});
  output_int.resize_({weight.size(0), input.size(0)});
  input_int.resize_({input.size(0), input.size(1)});
  
  // Call cuda kernel
  int M = weight.size(0);
  int K = weight.size(1); 
  int N = input.size(0);
  
  cutlass_linear_int4_kernel_wrapper(input.data<float>(),
				     input_int.data<signed char>(),
				     weight.data<int>(),
				     scale,
				     bias.data<float>(),
				     output_int.data<int>(),
				     output.data<float>(),
				     M, N, K);
}

void cutlass_linear_int1(torch::Tensor &input,
			 torch::Tensor &input_int,
			 torch::Tensor &weight,
			 float scale,
			 torch::Tensor &bias,
			 torch::Tensor &output_int,
			 torch::Tensor &output) {
  
  // Resize output if necessary
  output.resize_({weight.size(0), input.size(0)});
  output_int.resize_({weight.size(0), input.size(0)});
  input_int.resize_({input.size(0), input.size(1)});
  
  // Call cuda kernel
  int M = weight.size(0);
  int K = weight.size(1); 
  int N = input.size(0);
  
  cutlass_linear_int1_kernel_wrapper(input.data<float>(),
				     input_int.data<signed char>(),
				     weight.data<int>(),
				     scale,
				     bias.data<float>(),
				     output_int.data<int>(),
				     output.data<float>(),
				     M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_initialize", &cutlass_initialize);
  m.def("cutlass_linear_fp32", &cutlass_linear_fp32);
  m.def("cutlass_linear_fp16", &cutlass_linear_fp16);
  m.def("cutlass_linear_int8", &cutlass_linear_int8);
  m.def("cutlass_linear_int4", &cutlass_linear_int4);
  m.def("cutlass_linear_int1", &cutlass_linear_int1);  
}
