# coding: utf-8
import argparse
import numpy as np
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=2048,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

###################

parser.add_argument("--method", type=str, default="pbatch", choices=["cutlass","pbatch","fake"])
parser.add_argument("--W_bits", type=int, default=8)
parser.add_argument("--A_bits", type=int, default=8)

criterion = nn.CrossEntropyLoss()

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device("cuda")

###############################################################################
# Load data
###############################################################################

def Q(W, n):
    range = np.abs(np.min(W)) + np.abs(np.max(W))
    d = range / (2**(n))
    z = -np.min(W, 0) // d
    W = np.rint(W / d)
    W = d * (W)
    return W

def PQ_rounded(W, n):
    excl_zero = n < 3

    # Positive and negative
    W_sign = np.sign(W)
    W_fixed = (W * (2**16)).astype(int)
    max_place = int(np.ceil(np.log2(np.max(np.abs(W_fixed)))))+1
    mask = (2**max_place)-1
    mask = mask & ~(max(2**(max_place-n)-1, 0))
    bins = np.array([mask - x*2**(max_place-n) for x in range(2**n)])    
    if excl_zero:
        bins = bins[:-1]
    if len(bins) == 1:
        centers = bins
    else:
        centers = (bins[1:]+bins[:-1])/2
    rounded = bins[np.digitize(np.abs(W_fixed), centers)]
    rounded = (rounded * W_sign) / 2**16
    return rounded

def Q_opt(W, n):

    W_begin = PQ_rounded(W,n)
    W_begin_err = np.linalg.norm(W_begin-W)
    
    maximum = np.max(np.abs(W))
    sgn = np.sign(W)
    best_err, best_W = W_begin_err, W_begin
    for i in np.arange(.01, maximum, .01):
        cur = np.minimum(i, np.abs(W)) * sgn
        cur = PQ_rounded(cur, n)
        cur_err = np.linalg.norm(cur-W)
        if cur_err <= best_err:
            best_err = cur_err
            best_W = cur
    return best_W

corpus = data.Corpus(args.data)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 1
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            #sys.exit(0)
    return total_loss / (len(data_source) - 1)

def perf_bench(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(1)
    with torch.no_grad():
        data, targets = get_batch(data_source, 0)
        times = []        
        for i in range(10):
            torch.cuda.synchronize()
            tstart = time.time()
            for j in range(10):
                model(data, hidden)
            torch.cuda.synchronize()
            tend = time.time()
            times.append(tend-tstart)
        tt = min(times)
    return tt
        
if __name__ == "__main__":
    ntokens = len(corpus.dictionary)    
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, method=args.method,
                           W_bits=args.W_bits, A_bits=args.A_bits).to(device)
    
    # Load the best saved model.
    with open(args.save, 'rb') as f:
        state_dict = torch.load(f)
        #for k,v in state_dict.items():
        #    continue
        #    v_npy = v.cpu().numpy()
        #    if "rnn" in k and "bias" not in k:
        #        v_npy_q = PQ_rounded(v_npy, args.q)
        #    else:
        #        v_npy_q = v_npy
        #state_dict[k] = torch.from_numpy(v_npy_q)

    model.load_state_dict(state_dict, strict=False)
    model.eval()    

    test_ppl = math.exp(evaluate(test_data))
    perf = perf_bench(train_data)

    print("module,mnist,%s,%d,%d,%f,%f" % (args.method, args.W_bits, args.A_bits, test_ppl, perf))
