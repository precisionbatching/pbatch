#ifndef PBATCH
#define PBATCH

#include <iostream>
#include <vector>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

//#ifdef CUTLASS_USE_WMMA_API
//#ifdef CUTLASS_USE_SUBBYTE_WMMA
#pragma warning( disable : 4503)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#include "tools/test/perf/gemm/cutlass_dispatch.h"
#include "cutlass/gemm/sgemm_traits.h"

#define MAX(x,y) ((x)>(y)?(x):(y))
#define FIXED(x,act_shift_local) ((int)((x)*(act_shift_local)))

#define BLOCK_H 8
#define BLOCK_W 32
#define THREAD_H 2
#define THREAD_W 32
#define OUTER_PAR 32

struct PBatchLayer {
  cutlass::Vector<cutlass::bin1_t, 32> *layer;
  float scale;
};
typedef struct PBatchLayer PBatchLayer;

struct PBatchWeightLayers {
  std::vector<PBatchLayer> layers;
  int M, K;
  int *input_cvted;
};
typedef struct PBatchWeightLayers PBatchWeightLayers;

struct PBatchOutput {
  torch::Tensor result;
};
typedef struct PBatchOutput PBatchOutput;

//////////////////////////////////////////////////////////////

static int pbatch_initialized = 0;
void pbatch_initialize() {
  if (pbatch_initialized) return;
  pbatch_initialized = 1;
}

PBatchWeightLayers *pbatch_alloc_layers(int M, int K) {
  PBatchWeightLayers *w = new PBatchWeightLayers();
  w->M = M;
  w->K = K;
  cudaMalloc(&w->input_cvted, sizeof(int)*K*32);
  cudaMemset(w->input_cvted, 0, sizeof(int)*K*32);
  return w;
}

void pbatch_clear_layers(void *weight_layers) {
  PBatchWeightLayers *w = (PBatchWeightLayers *)weight_layers;
  w->layers.clear();
}

void pbatch_add_layer(void *weight_layers, int *layer, int M, int N, float scale) {
  PBatchWeightLayers *w = (PBatchWeightLayers *)weight_layers;
  int *weight_layer;
  cudaMalloc(&weight_layer, sizeof(int)*M*N);
  cudaMemcpy(weight_layer, layer, sizeof(int)*M*N, cudaMemcpyHostToDevice);
  w->layers.push_back((PBatchLayer){
      (cutlass::Vector<cutlass::bin1_t, 32> *)weight_layer, scale});       
}

void *pbatch_create_output(int M) {
  // Create intermediate arrays for computation  
  PBatchOutput *o = new PBatchOutput();
  auto options = torch::TensorOptions().
    dtype(torch::kFloat32).
    layout(torch::kStrided).
    device(torch::kCUDA, 0);
  o->result = torch::full({M, 1}, 0, options);
  return o;
}


//////////////////////////////////////////////////////////////////////
//                       MAIN KERNELS                               //
//////////////////////////////////////////////////////////////////////

///////////////////////////////////////////
// Bitwise matrix transpose on the input //
///////////////////////////////////////////
__global__ void pbatch_cvt_float_fixed_trans(int *dst, float *src, int K, int act_shift) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;  
  int id = block_id*blockDim.x + thread_id;

  int y = block_id; 
  int x = thread_id; 

  int v_1 = (FIXED(src[8*x+0],act_shift) >> y) & 1;
  int v_2 = (FIXED(src[8*x+1],act_shift) >> y) & 1;
  int v_3 = (FIXED(src[8*x+2],act_shift) >> y) & 1;
  int v_4 = (FIXED(src[8*x+3],act_shift) >> y) & 1;
  int v_5 = (FIXED(src[8*x+4],act_shift) >> y) & 1;
  int v_6 = (FIXED(src[8*x+5],act_shift) >> y) & 1;
  int v_7 = (FIXED(src[8*x+6],act_shift) >> y) & 1;
  int v_8 = (FIXED(src[8*x+7],act_shift) >> y) & 1;
  
  uint8_t v = ((v_1 << 7) |
	       (v_2 << 6) |
	       (v_3 << 5) |
	       (v_4 << 4) |
	       (v_5 << 3) |
	       (v_6 << 2) |
	       (v_7 << 1) |
	       (v_8 << 0));

  // Output shape: 32x(K/32)
  ((uint8_t *)dst)[y*K/8+x] = v;
}

////////////////////////////////
// NAIVE KERNEL FOR % 32 != 0 //
////////////////////////////////
__global__ void pbatch_kernel_naive(int *A, int *B, float *C, float *bias,
				    int M, int K, int N,
				    float alpha, float beta, float gamma,
				    int act_bits,
				    float act_shift_scale) {
  int block = blockIdx.x;
  int thread = threadIdx.x;
  
  __shared__ float C_shared[32];
  
  int *A_start = &A[block*K];
  int *B_start = &B[thread*K];
  float *C_start = &C_shared[thread];
  
  long long int sum = 0;
  for (int i = 0; i < K; i++) {
    int scale = ((long long int)1 << thread) * (thread == 31 ? -1 : 1);
    sum += __popc(A_start[i] & B_start[i])*scale;
  }
  C_start[0] = sum / (float)(act_shift_scale);
  
  __syncthreads();

  if (thread == 0) {
    float final = beta*C[block] + gamma*bias[block];
    for (int i = 0; i < act_bits; i++) {
      final += alpha*(float)C_shared[i];
    }
    C[block] = final;
  }
}

void pbatch_naive(PBatchWeightLayers *weight_layers,
		  float *input,
		  float *bias,
		  PBatchOutput *C,
		  int M, int K, int N,
		  int act_bits,
		  float act_shift_scale) {
  int *input_cvted = weight_layers->input_cvted;

  int blocks = M;
  int threads = 32;
  
  int nlayers = weight_layers->layers.size();
  pbatch_cvt_float_fixed_trans<<<32, K*32/8, 0, 0>>>
    (input_cvted, input, K*32, act_shift_scale);
  
  for (int i = 0; i < nlayers; i++) {
    float beta = i == 0 ? 0 : 1;
    float gamma = i == 0 ? 1 : 0;
    pbatch_kernel_naive<<<blocks, threads, 0, 0>>>
      ((int *)weight_layers->layers[i].layer,
       (int *)input_cvted,
       C->result.data<float>(),
       bias,
       M, K, 32,
       weight_layers->layers[i].scale, beta, gamma,
       act_bits, act_shift_scale);
  }  
}


//////////////////////////////////////
// Blocked Pbatch kernel (block=32) //
//////////////////////////////////////

__global__ void pbatch_inner_tiled_act16(int *A, int *B, float *C, float *bias,
					 int M, int K, int N,
					 float alpha, float beta, float gamma,
					 float act_shift_scale) {
  // Block row and column
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;
  
  // Thread row and column
  const int row = threadIdx.y;
  const int col = threadIdx.x;
  const int outer = threadIdx.z;

  // Within block id
  const int within_block_thread_id = row*(BLOCK_W/THREAD_W)*OUTER_PAR+col*OUTER_PAR+outer;

  __shared__ float Cs[BLOCK_H];
 
  if (within_block_thread_id < BLOCK_H) {
    Cs[within_block_thread_id] = 0;
  }

  __shared__ int accumC[OUTER_PAR][BLOCK_H];

  // Outer loop
  for (int k_outer = 0; k_outer < K; k_outer += BLOCK_W) {
    
    // Load to shared memory
    const int block_offset_A = blockRow*BLOCK_H*K + k_outer;
    const int block_offset_B = blockCol*BLOCK_W*K + k_outer;
        
    // Inner loop
    int Cl[THREAD_H] = {0};
    for (int k_inner = 0; k_inner < BLOCK_W; k_inner += THREAD_W) {

      // Assumes that each thread does a single outer product
      int Al[THREAD_H];
      int Bl[THREAD_W];
      for (int l = 0; l < THREAD_H; l++) {
	Al[l] = A[(row*THREAD_H+l)*K+(k_inner+outer)+block_offset_A];
      }
      for (int l = 0; l < THREAD_W; l++) {
	Bl[l] = B[(col*THREAD_W+l)*K+(k_inner+outer)+block_offset_B];
      }

      // Outer product
      int j_end = 16;
      for (int i = 0; i < THREAD_H; i++) {
	for (int j = 0; j < j_end; j++) {  
	  int glob_col_indx = j + col*BLOCK_W + k_inner;
	  int scale = (1 << glob_col_indx) * (j == j_end-1 ? -1 : 1);
	  Cl[i] += __popc(Al[i]&Bl[j])*scale;
	    
	}
	accumC[outer][i+THREAD_H*row] = Cl[i];
      }
    }
    
    // Write C to output
    __syncthreads();

    if (within_block_thread_id < BLOCK_H) {
      long long int sum = Cs[row*THREAD_H+outer];
      for (int i = 0; i < OUTER_PAR; i++) {
	sum += accumC[i][outer];
      }
      // Write to shared output
      Cs[row*THREAD_H+outer] += (float)sum / (act_shift_scale);
    }
    __syncthreads();
  }

  // Write to output
  if (within_block_thread_id < BLOCK_H) {
    const int indx = blockRow*BLOCK_H+within_block_thread_id;
    C[indx] = beta*C[indx] + alpha*(float)Cs[within_block_thread_id] + gamma*bias[indx];
  }
}

__global__ void pbatch_inner_tiled_act32(int *A, int *B, float *C, float *bias,
					 int M, int K, int N,
					 float alpha, float beta, float gamma,
					 float act_shift_scale) {
  // Block row and column
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;
  
  // Thread row and column
  const int row = threadIdx.y;
  const int col = threadIdx.x;
  const int outer = threadIdx.z;

  // Within block id
  const int within_block_thread_id = row*(BLOCK_W/THREAD_W)*OUTER_PAR+col*OUTER_PAR+outer;

  __shared__ float Cs[BLOCK_H];
 
  if (within_block_thread_id < BLOCK_H) {
    Cs[within_block_thread_id] = 0;
  }

  __shared__ int accumC[OUTER_PAR][BLOCK_H];

  // Outer loop
  for (int k_outer = 0; k_outer < K; k_outer += BLOCK_W) {
    
    // Load to shared memory
    const int block_offset_A = blockRow*BLOCK_H*K + k_outer;
    const int block_offset_B = blockCol*BLOCK_W*K + k_outer;
        
    // Inner loop
    int Cl[THREAD_H] = {0};
    for (int k_inner = 0; k_inner < BLOCK_W; k_inner += THREAD_W) {

      // Assumes that each thread does a single outer product
      int Al[THREAD_H];
      int Bl[THREAD_W];
      for (int l = 0; l < THREAD_H; l++) {
	Al[l] = A[(row*THREAD_H+l)*K+(k_inner+outer)+block_offset_A];
      }
      for (int l = 0; l < THREAD_W; l++) {
	Bl[l] = B[(col*THREAD_W+l)*K+(k_inner+outer)+block_offset_B];
      }

      // Outer product
      int j_end = 32;
      for (int i = 0; i < THREAD_H; i++) {
	for (int j = 0; j < j_end; j++) {  
	  int glob_col_indx = j + col*BLOCK_W + k_inner;
	  int scale = (1 << glob_col_indx) * (j == j_end-1 ? -1 : 1);
	  Cl[i] += __popc(Al[i]&Bl[j])*scale;
	    
	}
	accumC[outer][i+THREAD_H*row] = Cl[i];
      }
    }
    
    // Write C to output
    __syncthreads();

    if (within_block_thread_id < BLOCK_H) {
      long long int sum = Cs[row*THREAD_H+outer];
      for (int i = 0; i < OUTER_PAR; i++) {
	sum += accumC[i][outer];
      }
      // Write to shared output
      Cs[row*THREAD_H+outer] += (float)sum / (act_shift_scale);
    }
    __syncthreads();
  }

  // Write to output
  if (within_block_thread_id < BLOCK_H) {
    const int indx = blockRow*BLOCK_H+within_block_thread_id;
    C[indx] = beta*C[indx] + alpha*(float)Cs[within_block_thread_id] + gamma*bias[indx];
  }
}

__global__ void pbatch_inner_tiled_act8(int *A, int *B, float *C, float *bias,
					int M, int K, int N,
					float alpha, float beta, float gamma,
					float act_shift_scale) {
  // Block row and column
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;
  
  // Thread row and column
  const int row = threadIdx.y;
  const int col = threadIdx.x;
  const int outer = threadIdx.z;

  // Within block id
  const int within_block_thread_id = row*(BLOCK_W/THREAD_W)*OUTER_PAR+col*OUTER_PAR+outer;

  __shared__ float Cs[BLOCK_H];
 
  if (within_block_thread_id < BLOCK_H) {
    Cs[within_block_thread_id] = 0;
  }

  __shared__ int accumC[OUTER_PAR][BLOCK_H];

  // Outer loop
  for (int k_outer = 0; k_outer < K; k_outer += BLOCK_W) {
    
    // Load to shared memory
    const int block_offset_A = blockRow*BLOCK_H*K + k_outer;
    const int block_offset_B = blockCol*BLOCK_W*K + k_outer;
        
    // Inner loop
    int Cl[THREAD_H] = {0};
    for (int k_inner = 0; k_inner < BLOCK_W; k_inner += THREAD_W) {

      // Assumes that each thread does a single outer product
      int Al[THREAD_H];
      int Bl[THREAD_W];
      for (int l = 0; l < THREAD_H; l++) {
	Al[l] = A[(row*THREAD_H+l)*K+(k_inner+outer)+block_offset_A];
      }
      for (int l = 0; l < THREAD_W; l++) {
	Bl[l] = B[(col*THREAD_W+l)*K+(k_inner+outer)+block_offset_B];
      }

      // Outer product
      int j_end = 8;
      for (int i = 0; i < THREAD_H; i++) {
	for (int j = 0; j < j_end; j++) {  
	  int glob_col_indx = j + col*BLOCK_W + k_inner;
	  int scale = (1 << glob_col_indx) * (j == j_end-1 ? -1 : 1);
	  Cl[i] += __popc(Al[i]&Bl[j])*scale;
	    
	}
	accumC[outer][i+THREAD_H*row] = Cl[i];
      }
    }
    
    // Write C to output
    __syncthreads();

    if (within_block_thread_id < BLOCK_H) {
      long long int sum = Cs[row*THREAD_H+outer];
      for (int i = 0; i < OUTER_PAR; i++) {
	sum += accumC[i][outer];
      }
      // Write to shared output
      Cs[row*THREAD_H+outer] += (float)sum / (act_shift_scale);
    }
    __syncthreads();
  }

  // Write to output
  if (within_block_thread_id < BLOCK_H) {
    const int indx = blockRow*BLOCK_H+within_block_thread_id;
    C[indx] = beta*C[indx] + alpha*(float)Cs[within_block_thread_id] + gamma*bias[indx];
  }
}

void pbatch_inner_dispatch(dim3 &dimGrid,
			   dim3 &dimBlock,
			   PBatchWeightLayers *weight_layers,
			   int *input_cvted,
			   float *bias,
			   PBatchOutput *C,
			   int M, int K, int N,
			   int act_bits,
			   float act_shift_scale,   
			   int i, float beta, float gamma,
			   cudaStream_t stream) {
  if (act_bits == 16) {
    pbatch_inner_tiled_act16<<<dimGrid, dimBlock, 0, stream>>>
      ((int *)weight_layers->layers[i].layer,
       (int *)input_cvted,
       C->result.data<float>(),
       bias,
       M, K, 32,
       weight_layers->layers[i].scale, beta, gamma,
       act_shift_scale);    
  }
  if (act_bits == 32) {
    pbatch_inner_tiled_act32<<<dimGrid, dimBlock, 0, stream>>>
      ((int *)weight_layers->layers[i].layer,
       (int *)input_cvted,
       C->result.data<float>(),
       bias,
       M, K, 32,
       weight_layers->layers[i].scale, beta, gamma,
       act_shift_scale);    
  }
  if (act_bits == 8) {
    pbatch_inner_tiled_act8<<<dimGrid, dimBlock, 0, stream>>>
      ((int *)weight_layers->layers[i].layer,
       (int *)input_cvted,
       C->result.data<float>(),
       bias,
       M, K, 32,
       weight_layers->layers[i].scale, beta, gamma,
       act_shift_scale);    
  }
  
}

void pbatch_kernel_hblock32(PBatchWeightLayers *weight_layers,
			    float *input,
			    float *bias,
			    PBatchOutput *C,
			    int M, int K, int N,
			    int act_bits,
			    float act_shift_scale) {
  assert(N == 1);
  assert(OUTER_PAR == BLOCK_W);
  assert(BLOCK_W/THREAD_W*BLOCK_H/THREAD_H*OUTER_PAR <= 1024);
  assert(32/BLOCK_W > 0);
  assert(32/BLOCK_W == 1);
  assert(M/BLOCK_H > 0);

  int *input_cvted = weight_layers->input_cvted;
  dim3 dimGrid(32/BLOCK_W, M/BLOCK_H);
  dim3 dimBlock(BLOCK_W/THREAD_W, BLOCK_H/THREAD_H, OUTER_PAR);
  
  int nlayers = weight_layers->layers.size();
  pbatch_cvt_float_fixed_trans<<<32, K*32/8, 0, 0>>>
    (input_cvted, input, K*32, act_shift_scale);  
  
  for (int i = 0; i < nlayers; i++) {
    float beta = i == 0 ? 0 : 1;
    float gamma = i == 0 ? 1 : 0;
    pbatch_inner_dispatch(
			  dimGrid,
			  dimBlock,
			  weight_layers,
			  input_cvted,
			  bias,
			  C,
			  M, K, N,
			  act_bits,
			  act_shift_scale,
			  i, beta, gamma,
			  0);
  }  
}

// Kernel frontend 
void pbatch_kernel(PBatchWeightLayers *weight_layers,
		   float *input,
		   float *bias,
		   PBatchOutput *C,
		   int M, int K, int N,
		   int act_bits,
		   float act_shift_scale) {
  if (M % 32 == 0)
    pbatch_kernel_hblock32(weight_layers, input, bias, C, M, K, N,
			   act_bits, act_shift_scale);
  else 
    pbatch_naive(weight_layers, input, bias, C, M, K, N,
		 act_bits, act_shift_scale);
}

#endif
//#endif
//#endif
