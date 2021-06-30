
#ifndef CUTLASS_FP32
#define CUTLASS_FP32

#include "cutlass_header.h"

cudaError_t CutlassSgemmTN(
			   int M,
			   int N,
			   int K,
			   float alpha,
			   float const *A,
			   int lda,
			   float const *B,
			   int ldb,
			   float beta,
			   float *C,
			   int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size.
  //
  // Note, GemmTraits<> is a generic template defined for various general matrix product
  // computations within CUTLASS. It is intended to be maximally flexible, and consequently
  // it contains numerous template arguments.
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
  //
  typedef cutlass::gemm::SgemmTraits<
    cutlass::MatrixLayout::kRowMajor,   // layout of A matrix
    cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
    cutlass::Shape<8, 128, 128>            // threadblock tile size
    >
    GemmTraits;

  // Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

  // Construct and initialize CUTLASS GEMM parameters object.
  //
  // One of CUTLASS's design patterns is to define parameters objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  typename Gemm::Params params;

  int result = params.initialize(
				 M,     // GEMM M dimension
				 N,     // GEMM N dimension
				 K,     // GEMM K dimension
				 alpha, // scalar alpha
				 A,     // matrix A operand
				 lda,
				 B,     // matrix B operand
				 ldb,
				 beta,  // scalar beta
				 C,     // source matrix C
				 ldc,
				 C,     // destination matrix C (may be different memory than source C matrix)
				      ldc
				 );

  if (result) {
    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
    return cudaErrorInvalidValue;
  }

  // Launch the CUTLASS GEMM kernel.
  Gemm::launch(params);

  // Return any errors associated with the launch or cudaSuccess if no error.
  return cudaGetLastError();
}

void cutlass_linear_fp32_kernel(float *input,
				float *weight,
				float *bias,
				float *output,
				int M, int N, int K) {
  const float alpha = 1.0;
  const float beta = 1.0;
  
  cudaMemcpyAsync(output, bias, sizeof(float) * M * N, cudaMemcpyDeviceToDevice);
  
  CutlassSgemmTN(M, N, K,
		 alpha,
		 weight, K,
		 input, K, beta,
		 output, M);
}

#endif
