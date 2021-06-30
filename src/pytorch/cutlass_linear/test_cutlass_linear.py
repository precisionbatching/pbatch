import sys
import torch
import time
import numpy as np
import math
from cutlass_linear import CutlassLinear
from torch.nn.parameter import Parameter

def util_to_numpy(W):
    return W.cpu().detach().numpy()

def Q4(v):
    # 4 Bit quantization
    # Return scale and integer matrix representing quantized v
    v = util_to_numpy(v)
    maximum = np.max(np.abs(v))
    scale = 7.0/maximum

    # Unpack the 8-bit quantization of v
    weight_8bit = np.uint8(v*scale)
    unpacked_8bit = np.unpackbits(weight_8bit, axis=1)

    # We need to exclude the last 4-bits of 8-bit representation of weights
    def pack4bit(row):
        result = []
        for i in range(0, row.shape[0], 8):
            s = row[i]
            v = [s,s,s,s] + row[i+8-4:i+8].tolist()
            result += v
        return np.array(result, dtype=np.int8)

    trimmed_4bit = np.stack([pack4bit(unpacked_8bit[i]) for i in range(unpacked_8bit.shape[0])])
    packed = np.int32(np.packbits(trimmed_4bit, axis=1).view(dtype=np.int8))

    return packed, scale

def test_cutlass_fp32(M=1024,N=1024):
    W = CutlassLinear(N,M, quantize=32).cuda()    
    x = torch.from_numpy(np.random.randn(1, W.in_features)).float().cuda()
    
    correct = (util_to_numpy(W.weight).dot(util_to_numpy(x).T) +
               util_to_numpy(W.bias).reshape(-1, 1))
    predicted = util_to_numpy(W(x)).reshape(-1,1)    
    error = np.mean(np.abs(correct-predicted))
    
    assert(error <= 1e-5)

    print("Pass: test_cutlass_fp32 (M=%d,N=%d)" % (M,N))

def test_cutlass_fp16(M=1024,N=1024):
    W = CutlassLinear(N, M, quantize=16).cuda()
    x = torch.from_numpy(np.random.randn(1, W.in_features)).float().cuda()

    #W.weight = Parameter(torch.from_numpy(10*np.ones((1024,1024))).cuda())
    #W.bias = Parameter(torch.from_numpy(np.zeros((1024,1))).float().cuda())
    #W.prepare_quantization()
    
    correct = (util_to_numpy(W.weight.half()).dot(util_to_numpy(x).T) +
               util_to_numpy(W.bias).reshape(-1, 1))
    predicted = util_to_numpy(W(x)).reshape(-1,1)    
    error = np.mean(np.abs(correct-predicted))
    assert(error <= 1e-1) # Not sure why there is that much difference...

    print("Pass: test_cutlass_fp16 (M=%d,N=%d)" % (M,N))

def test_cutlass_int8(M=1024,N=1024):
    W = CutlassLinear(N, M, quantize=8).cuda()
    x = torch.from_numpy(np.random.randn(1, W.in_features)).float().cuda()

    #n_to_pad = math.ceil(N/256)*256 - N    
    #randommat = np.random.randn(M,N)
    #W.load_state_dict({".weight": torch.from_numpy(randommat),
    #                   ".bias": torch.from_numpy(np.zeros((M)))})

    #x_padded = torch.from_numpy(np.concatenate([util_to_numpy(x),
    #                                            np.zeros((1, n_to_pad))], axis=1)).float().cuda()
    
    
    # Quantize weights
    W_q = util_to_numpy(W.weight_int).astype(float)

    # Utilize cutlass_linear's quantize weights method to quantize input
    x_q, x_scale, _, _= W.quantize_weights(util_to_numpy(x), 8)
    x_q = util_to_numpy(x_q)

    # Compute scale = weight_scale * input_scale
    scale = W.scale*x_scale

    # Compute unscaled product
    unscaled = W_q.dot(x_q.T).astype(float)

    # Real answer = scaled_prod + bias
    correct = (unscaled/scale +
               util_to_numpy(W.bias).reshape(-1, 1))

    # Form prediction
    pred = util_to_numpy(W(x)).reshape(-1, 1)

    # Error checking
    error = np.mean(np.abs(pred-correct))
    
    assert(error <= 1e-5)

    print("Pass: test_cutlass_int8 (M=%d,N=%d)" % (M,N))

def test_cutlass_int4(M=1024,N=1024):
    W = CutlassLinear(N,M, quantize=4).cuda()
    x = torch.from_numpy(np.random.randn(1, W.in_features)).float().cuda()
        
    W_q, W_scale = Q4(W.weight)
    x_q, x_scale = Q4(x)

    # Compute unscaled product
    unscaled = W_q.dot(x_q.T).astype(float)
    scale = W_scale*x_scale

    # Real answer = scaled_prod + bias
    correct = (unscaled/scale +
               util_to_numpy(W.bias).reshape(-1, 1))

    # Form prediction
    pred = util_to_numpy(W(x)).reshape(-1, 1)

    # Error checking
    error = np.mean(np.abs(pred-correct))

    assert(error <= 1e-5)

    print("Pass: test_cutlass_int4 (M=%d,N=%d)" % (M,N))

def test_cutlass_int1(M=1024,N=1024):

    W = CutlassLinear(N,M, quantize=1).cuda()
    x = torch.from_numpy(np.random.randn(1, N)).float().cuda()

    #W.bias = Parameter(torch.from_numpy(np.zeros((1024,1))).float().cuda())
    #W.prepare_quantization()

    # 1-bit quantization = -1 or 1 (with scale = max of vector)
    W_q = util_to_numpy(W.weight)
    W_scale = np.max(np.abs(W_q))
    W_q = (-np.int32(W_q < 0) + np.int32(W_q >= 0))

    # "" for input
    x_q = util_to_numpy(x).reshape(-1, 1)
    x_scale = np.max(np.abs(x_q))
    x_q = (-np.int32(x_q < 0) + np.int32(x_q >= 0))

    n_to_pad = W_q.shape[-1]-x_q.shape[0]
    x_q = np.concatenate([x_q, np.zeros((n_to_pad, 1))])
    
    scale = W_scale*x_scale
    correct = scale*W_q.dot(x_q) + util_to_numpy(W.bias).reshape(-1, 1)
    pred = util_to_numpy(W(x)).reshape(-1, 1)

    error = np.mean(np.abs(pred-correct))
    assert(error <= 1e-5)    

    print("Pass: test_cutlass_int1 (M=%d,N=%d)" % (M,N))
    
if __name__=="__main__":
    tests = [
        {"test": test_cutlass_fp32,
         "args": 
         [
             {"M":1024,"N":1024},
             {"M":1024,"N":512},
             {"M":512,"N":1024},
             {"M":256,"N":4096},
             {"M":1152,"N":256},
             {"M":10,"N":256},
             {"M":2,"N":1024},
             {"M":8,"N":2},
             {"M":47,"N":24},
         ]},
        {"test": test_cutlass_fp16,
         "args": 
         [
             {"M":1024,"N":1024},
             {"M":1024,"N":512},
             {"M":512,"N":1024},
             {"M":256,"N":4096},
             {"M":1152,"N":256},
             {"M":10,"N":256},
             {"M":2,"N":1024},
             {"M":8,"N":2},
             {"M":47,"N":24},             
         ]},
        {"test": test_cutlass_int8,
         "args":
         [
             #{"M":1024, "N":784},
             {"M":1024,"N":1024},
             {"M":1024,"N":512},
             {"M":512,"N":1024},
             {"M":256,"N":4096},
             {"M":1152,"N":256},
             {"M":10,"N":256},
             {"M":10,"N":4096},
             {"M":2,"N":1024},
             {"M":8,"N":2},
             {"M":47,"N":24},
         ]},

        {"test": test_cutlass_int4,
         "args":
         [
             {"M":1024,"N":1024},
             {"M":1024,"N":512},
             {"M":512,"N":1024},
             {"M":256,"N":4096},
             {"M":1152,"N":256},
             {"M":10,"N":256},
             {"M":2,"N":1024},
             {"M":8,"N":2},
             {"M":47,"N":24},
         ]},
        {"test":test_cutlass_int1,
         "args":
         [
             # For int1, WMMA supports only sizes % 512 == 0
             {"M":8,"N":2},
             {"M":47,"N":24},
             {"M":1024,"N":1024},
             {"M":1024,"N":512},
             {"M":512,"N":1024},
             {"M":1024,"N":1536},
             {"M":10,"N":512},
         ]}
    ]
    for test in tests[::]:
        t = test["test"]
        args_list = test["args"]
        for kwargs in args_list:
            t(**kwargs)
        
