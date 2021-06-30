#ifndef CUTLASS_INT1
#define CUTLASS_INT1

#include "cutlass_header.h"

__global__ void quantize_int1(cutlass::Vector<cutlass::bin1_t, 32> *input_int,
			      float *input,       
			      int index_maximum,
			      int N) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;

  if (id < N) {
    float max_value = abs(input[index_maximum-1]);

    // Accumulate signs
    int r = 0;
    for (int i = 0; i < 32; i+=1) {
      int p1 = input[32*id+i] > 0;
      r <<= 1;
      r |= p1;
    }
    r = (r << 24) |
      ((r <<  8) & 0x00ff0000) |
      ((r >>  8) & 0x0000ff00) |
      ((r >> 24) & 0x000000ff);
    input_int[id] = r;
  }
}

__global__ void dequantize_int1(float *output,
				int *output_int,
				float *input, int index_maximum,
				float *bias,
				float weight_scale,
				int n_per_row,
				int N) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;

  if (id < N) {
    float input_scale = (float)1 / abs(input[index_maximum-1]);
    float scale = input_scale * weight_scale;
    int n_neg = output_int[id];
    int n_pos = n_per_row-n_neg;
    output[id] = bias[id] + (n_pos-n_neg)/scale;
    //output[id] = bias[id] + (n_pos-n_neg);
  }
}

cudaError_t CutlassInt1TN_1024(
			       int M,
			       int N,
			       int K,
			       char alpha,
			       cutlass::Vector<cutlass::bin1_t, 32> const *A,
			       int lda,
			       cutlass::Vector<cutlass::bin1_t, 32> const *B,
			       int ldb,
			       char beta,
			       int *C,
			       int ldc) {
  
  //assert(K*32 <= 1024);
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<512, 32, 32>,
    cutlass::Vector<cutlass::bin1_t, 32>,
    cutlass::Vector<cutlass::bin1_t, 32>,
    int,
    cutlass::gemm::LinearScaling<int>,
    int,
    cutlass::Shape<512, 16, 16>,
    cutlass::Shape<128, 8, 8>,
    128,
    128>
    WmmaGemmTraits;

  // Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
  typedef cutlass::gemm::Gemm<WmmaGemmTraits> Gemm;

  // Construct and initialize CUTLASS GEMM parameters object.
  typename Gemm::Params params;

  int result = params.initialize(
				 M,     // GEMM M dimension
				 N,     // GEMM N dimension
				 K,     // GEMM K dimension
				 alpha,   // scalar alpha
				 A,     // matrix A operand
				 lda,
				 B,     // matrix B operand
				 ldb,
				 beta,   // scalar beta
				 C,     // source matrix C
				 ldc,
				 C,     // destination matrix C (may be different memory than source C matrix)
				  M
				 );

  Gemm::launch(params);  
  return cudaGetLastError();
}

void cutlass_linear_int1_kernel(float *input,
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
  cublasIsamax(cutlass_handle, K*N*32, input, 1, &m);
  quantize_int1<<<max(1,K*N/N_THREADS), N_THREADS, 0, 0>>>((cutlass::Vector<cutlass::bin1_t, 32> *)input_int,
							   input, m, K*N);
  

  // 4-bit matmul
  CutlassInt1TN_1024(M, N, K*32,
		     alpha,
		     (cutlass::Vector<cutlass::bin1_t, 32> const *)weight, K,
		     (cutlass::Vector<cutlass::bin1_t, 32> const *)input_int, K, beta,
		     (int *)output_int, M);


  // Back to floating point
  dequantize_int1<<<max(1,M*N/N_THREADS), N_THREADS, 0, 0>>>(output,
							     output_int,
							     input, m,
							     bias,
							     scale,
							     K*32,
							     M*N);
}

#endif
