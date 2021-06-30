import torch
import json
import sys
import random
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

layer_id = 0

try:
    from cutlass_linear import CutlassLinear
    from pbatch_linear import PBatchLinear
except:
    pass

def Q(W, n):
    if n == 32:
        return W
    range = np.abs(np.min(W)) + np.abs(np.max(W))
    d = range / (2**(n))
    W = np.rint(W / d)
    W = d * (W)
    return W

def Q_tensor_slow(T, n):
    return torch.from_numpy(Q(T.detach().numpy(), n))

def Q_tensor(T, n):
    if n == 32:
        return T
    rng = torch.abs(torch.min(T)) + torch.abs(torch.max(T))
    d = (rng / 2**n + 1e-8)
    T_q = torch.round(T/d)*d

    # For sanity checking
    # assert(np.sum(Q_tensor_slow(T, n).detach().cpu().numpy()-T_q.detach().cpu().numpy()) == 0)

    return T_q

class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits=8):
        x = Q_tensor(x, bits)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, W_bits=8, A_bits=8, method="fake", optimize_cutoff=True, track_weights_acts=False, pbatch_act_shift=2**6):
        global layer_id
        super(QLinear, self).__init__(in_features, out_features, bias)
        assert(method in ["fake", "pbatch", "cutlass"])
        self.W_bits = W_bits
        self.A_bits = A_bits
        self.method = method
        self.layer_id = layer_id

        self.weight_values = []
        self.activation_values = []
        self.track_weights_acts = track_weights_acts

        layer_id += 1
        
        if method == "cutlass":
            assert(self.W_bits == self.A_bits)
            self.qmethod = CutlassLinear(in_features,out_features, bias=bias, quantize=self.W_bits, optimize_cutoff=optimize_cutoff).cuda()
        elif method == "pbatch":
            self.qmethod = PBatchLinear(in_features,out_features,bias=bias, W_bits=self.W_bits, A_bits=self.A_bits, optimize_cutoff=optimize_cutoff, act_shift=pbatch_act_shift).cuda()

    def forward(self, input):

        if self.method == "pbatch" or self.method == "cutlass":
            output = self.qmethod(input)
            return output
        
        qweight = FakeQuantOp.apply(self.weight, self.W_bits)
        output = F.linear(FakeQuantOp.apply(input, self.A_bits), qweight, self.bias)

        if self.track_weights_acts:
            self.weight_values = self.weight.detach().numpy()
            self.activation_values = input.detach().numpy()

        return output
