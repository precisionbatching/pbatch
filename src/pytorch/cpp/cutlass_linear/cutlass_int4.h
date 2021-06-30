#ifndef CUTLASS_INT4
#define CUTLASS_INT4

#include "cutlass_header.h"

__global__ void quantize_int4(cutlass::Vector<cutlass::int4_t, 8> *input_int,
			      float *input,       
			      int index_maximum,
			      int N) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;

  if (id < N) {
    float max_value = abs(input[index_maximum-1]);
    int v1 = ((int)(((float)7 / max_value) * input[8*id+0]) & 0xf);
    int v2 = ((int)(((float)7 / max_value) * input[8*id+1]) & 0xf);
    int v3 = ((int)(((float)7 / max_value) * input[8*id+2]) & 0xf);
    int v4 = ((int)(((float)7 / max_value) * input[8*id+3]) & 0xf);
    int v5 = ((int)(((float)7 / max_value) * input[8*id+4]) & 0xf);
    int v6 = ((int)(((float)7 / max_value) * input[8*id+5]) & 0xf);
    int v7 = ((int)(((float)7 / max_value) * input[8*id+6]) & 0xf);
    int v8 = ((int)(((float)7 / max_value) * input[8*id+7]) & 0xf);
    input_int[id] = (v1 << 0) |
      (v2 << 4) |
      (v3 << 8) |
      (v4 << 12) |
      (v5 << 16) |
      (v6 << 20) |
      (v7 << 24) |
      (v8 << 28);      
  }
}

__global__ void dequantize_int4(float *output,
				int *output_int,
				float *input, int index_maximum,
				float *bias,
				float weight_scale,
				int N) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;

  if (id < N) {
    float input_scale = (float)7 / abs(input[index_maximum-1]);
    float scale = input_scale * weight_scale;
    output[id] = bias[id] + output_int[id]/(float)scale;
  }
}

cudaError_t CutlassInt4TN(
			  int M,
			  int N,
			  int K,
			  char alpha,
			  cutlass::Vector<cutlass::int4_t, 8> const *A,
			  int lda,
			  cutlass::Vector<cutlass::int4_t, 8> const *B,
			  int ldb,
			  char beta,
			  int *C,
			  int ldc) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<256, 128, 128>,
    cutlass::Vector<cutlass::int4_t, 8>,
    cutlass::Vector<cutlass::int4_t, 8>,
    int,
    cutlass::gemm::LinearScaling<int>,
    int,
    cutlass::Shape<256, 32, 32>,
    cutlass::Shape<32, 8, 8>,
    32,
    32>
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

void cutlass_linear_int4_kernel(float *input,
				void *input_int,
				void *weight,
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
  cublasIsamax(cutlass_handle, K*N*8, input, 1, &m);
  quantize_int4<<<max(1,K*N/N_THREADS), N_THREADS, 0, 0>>>((cutlass::Vector<cutlass::int4_t, 8> *)input_int,
							   input, m, K*N);



  
  // 4-bit matmul
  CutlassInt4TN(M, N, K*8,
		alpha,
		(cutlass::Vector<cutlass::int4_t, 8> const *)weight, K,
		(cutlass::Vector<cutlass::int4_t, 8> const *)input_int, K, beta,
		(int *)output_int, M);

 

  // Back to floating point
  dequantize_int4<<<max(1,M*N/N_THREADS), N_THREADS, 0, 0>>>(output,
							     output_int,
							     input, m,
							     bias,
							     scale,
							     M*N);
}


#endif
