from qlinear import QLinear
import numpy as np
import time
import torch

if __name__=="__main__":
    in_features = 4096
    out_features = 4096


    """
    W_bits = 1
    A_bits = 8
    method = "pbatch"
    w = QLinear(in_features, out_features, W_bits=W_bits, A_bits=A_bits, method="pbatch", optimize_cutoff=False)
    x = torch.from_numpy(np.random.randn(1, in_features)).float().cuda()
    """

    #"""
    W_bits = 32
    A_bits = 32
    method = "cutlass"
    w = QLinear(in_features, out_features, W_bits=W_bits, A_bits=A_bits, method="cutlass", optimize_cutoff=False)
    x = torch.from_numpy(np.random.randn(1, in_features)).float().cuda()
    #"""

    # Warmup
    for i in range(10):
        w(x)
        
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(10000):
        w(x)
    torch.cuda.synchronize()
    print("%f" % (time.time()-tt))
