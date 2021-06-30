#ifndef CUTLASS_FP16
#define CUTLASS_FP16

#include "cutlass_header.h"

__global__ void quantize_fp16(half *input_half,
			      float *input,       
			      int N) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;
  
  if (id < N) {
    input_half[id] = __float2half(input[id]);
  }
}

__global__ void dequantize_fp16(float *output,
				half *output_half,
				float *bias,
				int N) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;
  
  if (id < N) {
    output[id] = bias[id] + __half2float(output_half[id]);
  }
}

cudaError_t CutlassFp16TN(
			  int M,
			  int N,
			  int K,
			  half alpha,
			  half const *A,
			  int lda,
			  half const *B,
			  int ldb,
			  half beta,
			  half *C,
			  int ldc) {
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,    
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 256, 128>,
    half,
    half,
    half,
    cutlass::gemm::LinearScaling<half>,
    half,
    cutlass::Shape<32, 64, 64>
    >
    WmmaGemmTraits;
  
  typedef cutlass::gemm::Gemm<WmmaGemmTraits> Gemm;
  
  typename Gemm::Params params;
  int result = params.initialize(
				 M,
				 N,
				 K,
				 alpha,
				 A,
				 lda,
				 B,
				 ldb,
				 beta,
				 C,
				 ldc,
				 C,
				 ldc);
  if (result) {
    std::cerr << "Failed to initialized Cutlass Wmma Gemm fp16 object." << std::endl;
    return cudaErrorInvalidValue;
  }
  
  Gemm::launch(params);
  
  return cudaGetLastError();
}

void cutlass_linear_fp16_kernel(float *input,
				half *input_half,
				half *weight,
				float *bias,
				half *output_half,
				float *output,
				int M, int N, int K) {
  
#define N_THREADS 128
  
  float alpha = 1.0;
  float beta = 0;
  
  // Quantize input
  quantize_fp16<<<max(1,K*N/N_THREADS), N_THREADS, 0, 0>>>(input_half,
							   input, K*N);
  
  
  // 16-bit matmul
  CutlassFp16TN(M, N, K,
		alpha,
		weight, K,
		input_half, K, beta,
		output_half, M);
  
  // Back to floating point
  dequantize_fp16<<<max(1,M*N/N_THREADS), N_THREADS, 0, 0>>>(output,
							     output_half,
							     bias,
							     M*N);
}

#endif
