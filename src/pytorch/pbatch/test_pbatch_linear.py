import sys
import torch
import time
import numpy as np
import math
from pbatch_linear import PBatchLinear
from torch.nn.parameter import Parameter

def reference(W, x):

    # reference currently only works well when validating
    # act bits == 32 and act_shift == 16
    assert(W.A_bits == 32 and W.act_shift == 2**16)

    # Extract weight layers and pack
    weight_layers_npy = [(np.unpackbits(
        x[0].view(dtype=np.uint8), axis=1).astype(np.int), x[1]) for
                           x in W.bitlayers]
    
    # Extract input and pack
    x_npy = x.cpu().numpy()
    x_npy = (x_npy * (1<<16)).astype(int)
    inp_unpacked = np.unpackbits(np.array(x_npy, dtype='>i8').
                                 view(np.uint8)).reshape(-1, 64)[:,31:-1]
    inp_unpacked = inp_unpacked.astype(int)

    # Create output
    s = np.zeros((weight_layers_npy[0][0].shape[0], inp_unpacked.shape[-1]), dtype=float)

    # Accumulate
    for weight_layer, weight_scale in weight_layers_npy:
        inp = np.zeros((weight_layer.shape[-1], 32))
        inp[:inp_unpacked.shape[0],:inp_unpacked.shape[1]] = inp_unpacked
        s += weight_scale * weight_layer.dot(inp)
    inp_scale = [(-1 if i==0 else 1)*(pow(2, 16-i)) for i in range(0, 32)]
    res = s.dot(inp_scale)
    return res

def test_pbatch(M=1024,N=1024,W_bits=8,A_bits=8,act_shift=16):
    W = PBatchLinear(N, M, W_bits=W_bits, A_bits=A_bits, act_shift=act_shift).cuda()
    x = torch.from_numpy(np.random.randn(1, N)).float().cuda()

    pred = W(x).cpu().numpy().flatten()
    ref = reference(W, x).flatten()

    error = np.mean(np.abs(pred-ref))
    
    if error >= 1e-1:
        print("Fail: error=%f" % error)
        sys.exit(1)

    print("Pass: test_pbatch (M=%d,N=%d,W_bits=%d,A_bits=%d,act_shift=%d) err=%f" % (
        M,N,W_bits,A_bits,act_shift, error))

if __name__=="__main__":
    tests = [
        {
            "test" : test_pbatch,
            "args" : [                
                {"M":1024,"N":32,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":512,"N":64,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":512,"N":256,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":32,"N":1234,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":32,"N":1024,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":1024,"N":1024,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":1024,"N":4096,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":4096,"N":4096,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":1024,"N":4096,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":128,"N":4096,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":64,"N":4096,"W_bits":8,"A_bits":32,"act_shift":2**16},
                {"M":32,"N":1024,"W_bits":8,"A_bits":32,"act_shift":2**16},                
            ]
        }
    ]
    for test in tests:
        t = test["test"]
        args_list = test["args"]
        for kwargs in args_list:
            t(**kwargs)
        
