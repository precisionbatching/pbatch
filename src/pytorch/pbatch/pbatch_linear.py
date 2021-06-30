import torch
import sys
import time
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import itertools
import math
from pathos.multiprocessing import ProcessingPool as Pool

import pbatch_cuda

def compute_quantization_error(W, W_original, n_bits, scale=2**16):
    quantized, max_bit = pbatch_round_quantize(W, n_bits, scale=scale)
    approx = quantized / scale
    approx_err = np.mean(np.abs(approx-W_original))
    return approx_err

def compute_thresholded_quantization_error(W_original, threshold, n_bits, scale=2**16):
    sgn = np.sign(W_original)
    thresholded = sgn*np.minimum(threshold, np.abs(W_original))
    err = compute_quantization_error(thresholded, W_original, n_bits)
    return threshold, err

def pbatch_round_quantize(W, n, scale=2**16):
    #excl_zero = n < 4
    excl_zero = n == 1
    W_sign = np.sign(W).astype(int)
    W_fixed = (W * (scale)).astype(int)
    try:
        max_place = int(np.ceil(np.log2(np.max(np.abs(W_fixed)))))+1
        max_place = max(max_place, n)
    except:
        max_place = 16
    mask = (2**max_place)-1
    mask = mask & ~(max(2**(max_place-n)-1, 0))
    bins = np.array([mask - x*2**(max_place-n) for x in range(2**n)])
    if excl_zero:
        bins = bins[:-1]
    if len(bins) == 1:
        centers = bins
    else:
        centers = (bins[1:]+bins[:-1])/2

    rounded = (
        bins[np.minimum(len(centers)-1, np.digitize(np.abs(W_fixed), centers))] *
        W_sign)
    return rounded, max_place

class PBatchLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, W_bits=1,
                 A_bits=32, act_shift=2**6, optimize_cutoff=False,
                 n_opt_intervals=100):
        super(PBatchLinear, self).__init__()
        pbatch_cuda.pbatch_initialize()

        self.optimize_cutoff = optimize_cutoff
        self.A_bits = A_bits
        self.act_shift = act_shift
        self.n_opt_intervals = n_opt_intervals

        assert(self.A_bits in [32, 16, 12, 8])

        # Round up in_features to a multiple of 32
        in_features_padded = math.ceil(in_features/1024)*1024
        self.n_to_pad = in_features_padded-in_features
        self.in_features = in_features_padded
        self.out_features = out_features
        self.W_bits = W_bits
        self.weight = Parameter(torch.Tensor(out_features, self.in_features))
        self.output = pbatch_cuda.pbatch_create_output(out_features)
        self.pbatch_weights = pbatch_cuda.pbatch_alloc_weight_layers(out_features,
                                                                     self.in_features//32)
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.initialized = 0            
        self.reset_parameters()

    #################################################################
    def testing(self, test):
        return test+1

    def optimize_cutoff_W(self, n_processes=4):
        W_original = self.weight.detach().cpu().numpy()        

        W_best, err = W_original, compute_quantization_error(W_original, W_original, self.W_bits)
        maximum, minimum = np.max(np.abs(W_original)), np.min(np.abs(W_original))

        with Pool(n_processes) as p:
            a1, a2, a3 = [], [], []
            for threshold in np.arange(minimum, maximum,  max((maximum-minimum)/self.n_opt_intervals, .001)):
                a1.append(W_original)
                a2.append(threshold)
                a3.append(self.W_bits)
            results = p.map(compute_thresholded_quantization_error, a1,a2,a3)
            best_threshold, err = results[np.argmin([x[1] for x in results])]
            
            sgn = np.sign(W_original)
            W_best= sgn*np.minimum(best_threshold, np.abs(W_original))

        print("Done. Weight matrix approximation error: %s" % (err))

        return W_best
    
    def initialize_bitlayers(self):
        self.initialized += 1

        # Optimize cutoff
        if self.optimize_cutoff:
            W = self.optimize_cutoff_W()
        else:
            W = self.weight.detach().cpu().numpy()

        # Decompose weight layers
        self.bitlayers = (
            self.decompose_weight_layers(W, self.W_bits))        
        
        # Clear and initialize from decomposed layers
        pbatch_cuda.pbatch_clear_layers(self.pbatch_weights)
        for layer, scale in self.bitlayers:
            pbatch_cuda.pbatch_add_weight_layer(self.pbatch_weights,
                                                torch.from_numpy(layer).int(),
                                                scale)

    def pbatch_round_quantize(self, W, n, scale=2**16):
        return pbatch_round_quantize(W, n, scale)

    def extract_bitplane(self, W, ind):
        res = (np.bitwise_and(W, 1<<ind) != 0).astype(int)
        return res

    def packbits(self, m):
        return np.packbits(m, axis=1).view(dtype=np.int32)
            
    def decompose_weight_layers(self, A, n, scale=2**16):
        assert(n <= 16)
        quantized, max_bit = self.pbatch_round_quantize(A, n, scale=scale)
        unique_values = list(set(quantized.flatten()))
        assert(len(unique_values) <= 2**n+1)
        layers = [(self.extract_bitplane(quantized, x), (1<<x) *
                   (-1 if x == max_bit else 1)) for x in range(max_bit-n, max_bit+1)]
        layers = [(x[0], x[1]/scale) for x in layers]
        return [(self.packbits(x[0]), x[1]) for x in layers]

    #######################################################
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
           fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
           bound = 1 / math.sqrt(fan_in)
           init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = torch.from_numpy(np.zeros(self.bias.size())).float().cuda()

        self.weight = Parameter(torch.from_numpy(np.random.randn(self.out_features, self.in_features)).float().cuda())
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        # For the target weight, pad with zeros if shape mismatch
        prefix = prefix.split(".")[0] + "."
        weight_key = prefix + "weight"
        weight_npy = np.zeros(self.weight.size())
        state_dict_npy = state_dict[weight_key].detach().cpu().numpy()
        weight_npy[:state_dict_npy.shape[0],:state_dict_npy.shape[1]] = state_dict_npy        
        state_dict[weight_key] = torch.from_numpy(weight_npy)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.initialize_bitlayers()        

    def forward(self, input):

        if self.initialized < 1:
            self.initialize_bitlayers()
        pbatch_cuda.pbatch(self.pbatch_weights,
                           input,
                           self.bias,                           
                           self.output,
                           self.A_bits,
                           self.act_shift)
        return pbatch_cuda.pbatch_fetch_result(self.output).view(1, -1)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )    

if __name__=="__main__":
    # Basic sanity check on 32-bit, 1024x1024x1 matmul
    d1, d2 = 1024, 1024
    W = PBatchLinear(d2, d1, W_bits=4, A_bits=8).cuda()
    x = torch.from_numpy(np.random.randn(1, d2)).float().cuda()
    print(W(x))

