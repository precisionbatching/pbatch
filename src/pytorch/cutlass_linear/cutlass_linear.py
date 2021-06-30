import sys
import torch
import time
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
from pathos.multiprocessing import ProcessingPool as Pool

import cutlass_cuda

def compute_quantization_error(W, W_original, bits):
    assert bits in [1,4,8];
    if bits == 1:
        maximum = np.max(np.abs(W))
        scale = 1.0 / maximum
        sign = W >= 0;
        approx = np.float32(sign)/scale
    elif bits == 4:
        maximum = np.max(np.abs(W))
        scale = 7.0 / maximum
        weight_raw = np.int8(W*scale)
        weight_raw = weight_raw & 0xf
        weight_raw |= ((weight_raw >> 3)&1)*0xf0

        approx = np.float32(weight_raw)/scale
    elif bits == 8:
        maximum = np.max(np.abs(W))
        scale = 127.0 / maximum
        approx = np.float32(np.int8(W*scale))/scale

    return np.mean(np.abs(W_original-approx))

def compute_thresholded_quantization_error(W_original, threshold, n_bits, scale=2**16):
    sgn = np.sign(W_original)
    thresholded = sgn*np.minimum(threshold, np.abs(W_original))
    err = compute_quantization_error(thresholded, W_original, n_bits)
    return threshold, err

class CutlassLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, 
                 bias=True, quantize=32,
                 optimize_cutoff=False, n_opt_intervals=100):

        # Initialize
        super(CutlassLinear, self).__init__()
        assert(quantize in [16, 32, 8, 4, 1])
        cutlass_cuda.cutlass_initialize()

        # Special case for int1
        if quantize == 1:
            # Why we need to do this:
            # Tensorcore 1-bit MMA enforces a block size of 512.
            # To make it work, we need 512-in_features to cancel each other out.
            # We do this by setting each extra weight value alternating -1, +1 to cancel out.
            # Only need to do this for int1 as others can represent 0.
            assert(in_features % 2 == 0)
            self.n_padded = math.ceil(in_features/512)*512 - in_features
            self.in_features = math.ceil(in_features/512)*512
            
        else:
            self.n_padded = math.ceil(in_features/256)*256 - in_features
            self.in_features = math.ceil(in_features/256)*256
        self.out_features = out_features
        self.quantize = quantize
        self.optimize_cutoff = optimize_cutoff
        self.n_opt_intervals = n_opt_intervals

        # Weights to use and parameters
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.output = torch.Tensor(1, 1).cuda()
        
        # For quantization 
        self.scale = None
        self.weight_int = None
        self.input_int = None
        self.output_int = None

        # Bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Init
        self.reset_parameters()

    def reset_parameters(self):
       init.kaiming_uniform_(self.weight, a=math.sqrt(5))

       if self.bias is not None:
           fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
           bound = 1 / math.sqrt(fan_in)
           init.uniform_(self.bias, -bound, bound)

       self.prepare_quantization()

    def quantize_weights(self, W, bits):

        if bits == 16:
            weight = torch.from_numpy(W).half().cuda()
            input_int = torch.Tensor(1, 1).half().cuda()
            output_int = torch.Tensor(1, 1).half().cuda()
            scale = None
        elif bits == 8:
            maximum = np.max(np.abs(W))
            scale = 127.0 / maximum
            weight = torch.from_numpy(W*scale).char().cuda()
            input_int = torch.Tensor(1, 1).char().cuda()
            output_int = torch.Tensor(1, 1).int().cuda()            
        elif bits == 4:
            maximum = np.max(np.abs(W))
            scale = 7.0 / maximum
            
            # Pack 4 bits per 32-bit int
            weight_raw = np.uint8(W*scale)
            def packrow(row):
                
                result = np.zeros(int(row.shape[0]/2), dtype=np.int32)
                for i in range(0, row.shape[0], 16):                    
                    s1 = int(i/8)
                    s2 = int((i+8)/8)
                    
                    result[s2*4:s2*4+4] = row[i+8-4:i+8]                    
                    result[s1*4:s1*4+4] = row[i+8+8-4:i+8+8]                    
                return result

            def packrow2(row):
                result = []
                for i in range(0, row.shape[0], 8):
                    s = row[i]
                    v = [s,s,s,s] + row[i+8-4:i+8].tolist()
                    result += v
                return np.array(result, dtype=np.int8)
            
            unpacked = np.unpackbits(weight_raw, axis=1)
            trimmed = np.stack([packrow(unpacked[i]) for i in range(unpacked.shape[0])])
            packed = np.packbits(trimmed, axis=1).view(dtype=np.int32)            
            weight = torch.from_numpy(packed).int().cuda()                                   
            input_int = torch.Tensor(1, 1).char().cuda()
            output_int = torch.Tensor(1, 1).int().cuda()
        elif bits == 1:
            strt = self.in_features-self.n_padded
            end = strt+self.n_padded
            W[:,strt:end:2] = 1
            W[:,strt+1:end:2] = -1
                        
            maximum = np.max(np.abs(W))
            scale = 1.0 / maximum
            sign = W >= 0;
            packed = np.packbits(np.int8(sign), axis=1).view(dtype=np.int32)
            weight = torch.from_numpy(packed).int().cuda()
            input_int = torch.Tensor(1, 1).char().cuda()
            output_int = torch.Tensor(1, 1).int().cuda()
        else:
            assert(0)
            
        return weight, scale, input_int, output_int

    def optimize_cutoff_W(self, n_processes=4):        
        W_original = self.weight.detach().cpu().numpy()

        if self.quantize == 16 or self.quantize == 32:
            return W_original

        W_best, err = W_original, compute_quantization_error(W_original, W_original, self.quantize)
        maximum, minimum = np.max(np.abs(W_original)), np.min(np.abs(W_original))

        with Pool(n_processes) as p:
            a1, a2, a3 = [], [], []
            for threshold in np.arange(minimum, maximum,  max((maximum-minimum)/self.n_opt_intervals, .001)):
                a1.append(W_original)
                a2.append(threshold)
                a3.append(self.quantize)
            results = p.map(compute_thresholded_quantization_error, a1,a2,a3)
            best_threshold, err = results[np.argmin([x[1] for x in results])]
            
            sgn = np.sign(W_original)
            W_best= sgn*np.minimum(best_threshold, np.abs(W_original))

        print("Done. Weight matrix approximation error: %s" % (err))

        return W_best

    def prepare_quantization(self):
       if self.optimize_cutoff:
           weight = self.optimize_cutoff_W()
       else:
           weight = self.weight.detach().cpu().numpy()
       if self.quantize in [1, 4, 8, 16]:
           self.weight_int, self.scale, self.input_int, self.output_int = (
               self.quantize_weights(weight, self.quantize))        
        
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
        self.prepare_quantization()
        
    def get_forward_parameters(self, input):
        # Set parameters for quantization
        if self.quantize == 32:
            return (input, self.weight.data, self.bias.data, self.output)
        elif self.quantize == 16:
            return (input, self.input_int, self.weight_int, 
                    self.bias.data, self.output_int, self.output)   
        elif self.quantize == 8:
            return (input, self.input_int, self.weight_int.data, self.scale,
                    self.bias.data, self.output_int, self.output)
        elif self.quantize == 4:
            return (input, self.input_int, self.weight_int.data, self.scale,
                    self.bias.data, self.output_int, self.output)
        elif self.quantize == 1:
            n_to_pad = self.n_padded
            if n_to_pad != 0:
                # Special padding case for int1
                input = F.pad(input=input, pad=(0, n_to_pad), mode="constant", value=0)
            return (input, self.input_int, self.weight_int.data, self.scale,
                    self.bias.data, self.output_int, self.output)        
        else:
            assert(0)

    def get_forward_method(self):
        if self.quantize == 32:
            return cutlass_cuda.cutlass_linear_fp32
        elif self.quantize == 16:
            return cutlass_cuda.cutlass_linear_fp16
        elif self.quantize == 8:
            return cutlass_cuda.cutlass_linear_int8
        elif self.quantize == 4:
            return cutlass_cuda.cutlass_linear_int4
        elif self.quantize == 1:
            return cutlass_cuda.cutlass_linear_int1                
        else:
            assert(0)
           
    def forward(self, input):
       method = self.get_forward_method()
       params = self.get_forward_parameters(input)       
       method(*params)
       return self.output.view(1, -1)


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, q={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.quantize
        )
    

if __name__=="__main__":

    # Basic sanity check on 32-bit, 1024x1024x1 matmul
    q = 32
    d1, d2 = 1024, 1024
    W = CutlassLinear(d2, d1, quantize=q).cuda()
    x = torch.from_numpy(np.random.randn(1, d2)).float().cuda()
    print(W(x))
