from qlinear import QLinear
import sys
import numpy as np
import time
import torch

if __name__=="__main__":

    in_features, out_features, method, W_bits, A_bits = sys.argv[1:]
    in_features, out_features = int(in_features), int(out_features)
    W_bits, A_bits = int(W_bits), int(A_bits)    

    if method == "pbatch":
        w = QLinear(in_features, out_features, W_bits=W_bits, A_bits=A_bits, method="pbatch", optimize_cutoff=False)
        x = torch.from_numpy(np.random.randn(1, in_features)).float().cuda()

    if method == "cutlass":
        w = QLinear(in_features, out_features, W_bits=W_bits, A_bits=A_bits, method="cutlass", optimize_cutoff=False)
        x = torch.from_numpy(np.random.randn(1, in_features)).float().cuda()

    # Warmup
    for i in range(10):
        w(x)
        
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(10000):
        w(x)
    torch.cuda.synchronize()
    print("%f" % (time.time()-tt))
