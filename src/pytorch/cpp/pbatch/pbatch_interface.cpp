// Functions called in pytorch code.
  
#include <torch/extension.h>
#include <stdio.h>
#include <vector>

/***********************
 * External Functions  *
 ***********************/

struct PBatchWeightLayers;
struct PBatchOutput;

void pbatch_initialize();
PBatchWeightLayers *pbatch_alloc_layers(int M, int K);
void pbatch_kernel_wrapper(void *weight_layers,
			   float *input,
			   float *bias,
			   void *C,
			   int act_bits,
			   float act_shift_scale);
void pbatch_add_layer(void *weight_layers, int *layer, int M, int N, float scale);
void pbatch_clear_layers(void *weight_layers);
void *pbatch_create_output(int M);
torch::Tensor &pbatch_fetch_result_wrapper(void *output);

void pbatch(void *weight_layers,
	    torch::Tensor &input,
	    torch::Tensor &bias,
	    void *output,
	    int act_bits,
	    float act_shift_scale) {
  assert(input.size(0) == 1);
  pbatch_kernel_wrapper(weight_layers,
			input.data<float>(),
			bias.data<float>(),
			output,
			act_bits,
			act_shift_scale);
}

void pbatch_add_weight_layer(void *weight_layers,
			     torch::Tensor &layer,
			     float scale) {
  pbatch_add_layer(weight_layers, layer.data<int>(),
		   layer.size(0), layer.size(1), scale);
}

void *pbatch_alloc_weight_layers(int M, int K) {
  return (void *)pbatch_alloc_layers(M, K);
}


torch::Tensor &pbatch_fetch_result(void *output) {
  return pbatch_fetch_result_wrapper((PBatchOutput *)output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pbatch_initialize", &pbatch_initialize);
  m.def("pbatch_clear_layers", &pbatch_clear_layers);  
  m.def("pbatch", &pbatch);  
  m.def("pbatch_add_weight_layer", &pbatch_add_weight_layer);
  m.def("pbatch_alloc_weight_layers", &pbatch_alloc_weight_layers);
  m.def("pbatch_create_output", &pbatch_create_output);
  m.def("pbatch_fetch_result", &pbatch_fetch_result);
}
