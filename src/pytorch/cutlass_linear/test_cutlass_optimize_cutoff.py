import sys
import torch
import time
import numpy as np
import math
from cutlass_linear import CutlassLinear
from torch.nn.parameter import Parameter

def util_to_numpy(W):
    return W.cpu().detach().numpy()

def test_cutlass_optimize_cutoff(M=1024,N=1024,q=8):
    W = CutlassLinear(N, M, quantize=q, optimize_cutoff=True).cuda()
    x = torch.from_numpy(np.random.randn(1, N)).float().cuda()

    tstart = time.time()
    W.prepare_quantization()
    tend = time.time()
    telaps = tend-tstart

    print("Done. M=%d,N=%d,q=%d telapsed=%f" % (
        M,N,q,telaps))
    
if __name__=="__main__":
    tests = [
        {"test": test_cutlass_optimize_cutoff,
         "args": 
         [
             {"M":4096,"N":4096,"q":8},
             {"M":4096,"N":4096,"q":4},
             {"M":4096,"N":4096,"q":1},
         ]}
    ]
    for test in tests:
        t = test["test"]
        args_list = test["args"]
        for kwargs in args_list:
            t(**kwargs)
        
