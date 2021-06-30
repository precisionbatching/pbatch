#ifndef CUTLASS_INT8
#define CUTLASS_INT8

#include "cutlass_header.h"

__global__ void quantize_int8(signed char *input_int,
			      float *input,       
			      int index_maximum,
			      int N) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;
  
  if (id < N) {
    float max_value = abs(input[index_maximum-1]);
    signed char value = (signed char)(((float)127 / max_value) * input[id]);
    input_int[id] = value;
  }
}

__global__ void dequantize_int8(float *output,
				int *output_int,
				float *input, int index_maximum,
				float *bias,
				float weight_scale,
				int N) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;
  
  if (id < N) {
    float input_scale = (float)127 / abs(input[index_maximum-1]);
    float scale = input_scale * weight_scale;
    output[id] = bias[id] + output_int[id]/(float)scale;
  }
}

cudaError_t CutlassInt8TN(
			  int M,
			  int N,
			  int K,
			  int alpha,
			  signed char const *A,
			  int lda,
			  signed char const *B,
			  int ldb,
			  int beta,
			  int *C,
			  int ldc) {
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,      // layout of A matrix
    cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
    cutlass::Shape<128, 128, 128>,         // threadblock tile size
    signed char,                           // A type
    signed char,                           // B type
    int,                                   // D type
    cutlass::gemm::LinearScaling<int>,     // functor to do the math in the epilogue
    int,                                   // accumulator type
    cutlass::Shape<128, 32, 32>,           // warp tile size
    cutlass::Shape<16, 16, 16>,            // WMMA instruction tile size
    16,                                    // scalars every time a thread loads from A
    16                                     // scalars every time a thread loads from B
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
    std::cerr << "Failed to initialized Cutlass Wmma Gemm int8 object." << std::endl;
    return cudaErrorInvalidValue;
  }
  
  Gemm::launch(params);
  
  return cudaGetLastError();
}

void cutlass_linear_int8_kernel(float *input,
				signed char *input_int,
				signed char *weight,
				float scale,
				float *bias,
				int *output_int,
				float *output,
				int M, int N, int K) {
  
#define N_THREADS 128
  
  const float alpha = 1.0;
  const float beta = 0.0;  
  
  // Quantize input
  int m;
  cublasIsamax(cutlass_handle, K*N, input, 1, &m);
  quantize_int8<<<max(1,K*N/N_THREADS), N_THREADS, 0>>>(input_int, input, m, K*N);
  
  // 8-bit matmul
  CutlassInt8TN(M, N, K,
		alpha,
		(signed char const *)weight, K,
		(signed char const *)input_int, K, beta,
		(int *)output_int, M);
  
  // Back to floating point
  dequantize_int8<<<max(1,M*N/N_THREADS), N_THREADS, 0>>>(output,
							  output_int,
							  input, m,
							  bias,
							  scale,
							  M*N);
}


#endif
