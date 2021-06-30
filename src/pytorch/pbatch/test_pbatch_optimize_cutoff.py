import sys
import torch
import time
import numpy as np
import math
from pbatch_linear import PBatchLinear
from torch.nn.parameter import Parameter

def test_pbatch_optimize_cutoff(M=1024, N=1024, W_bits=8, A_bits=8, act_shift=16):
    W = PBatchLinear(N, M, W_bits=W_bits, A_bits=A_bits, act_shift=act_shift, optimize_cutoff=True).cuda()
    x = torch.from_numpy(np.random.randn(1, N)).float().cuda()

    tstart = time.time()
    W.initialize_bitlayers()
    tend = time.time()
    telapsed = tend-tstart

    print("Done. M=%d,N=%d,W_bits=%d,A_bits=%d,act_shift=%d, elapsed:%f" % (
        M,N,W_bits,A_bits,act_shift,telapsed))

if __name__=="__main__":
    tests = [
        {
            "test" : test_pbatch_optimize_cutoff,
            "args" : [
                {"M":4096,"N":4096,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":1024,"N":32,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":512,"N":64,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":512,"N":256,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":32,"N":1234,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":32,"N":1024,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":1024,"N":1024,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":1024,"N":4096,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":1024,"N":4096,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":128,"N":4096,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":64,"N":4096,"W_bits":8,"A_bits":32,"act_shift":16},
                {"M":32,"N":1024,"W_bits":8,"A_bits":32,"act_shift":16},                
            ]
        }
    ]
    for test in tests:
        t = test["test"]
        args_list = test["args"]
        for kwargs in args_list:
            t(**kwargs)
        
