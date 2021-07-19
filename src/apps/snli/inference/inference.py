import os
import sys
import numpy as np
import json
import dill as pickle
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from torchtext.legacy import data
from torchtext.legacy import datasets

from model import SNLIClassifier
from util import get_args, makedirs


args = get_args()
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')

inputs = data.Field(lower=args.lower, tokenize='spacy')
answers = data.Field(sequential=False)

if os.path.exists("train.json"):
    fields = {'premise': ('premise', inputs),
              'hypothesis': ('hypothesis', inputs),
              'label': ('label', answers)}
    train, dev, test = data.TabularDataset.splits(
        path = '',
        train = 'train.json',
        validation='dev.json',
        test = 'test.json',
        format = 'json',
        fields=fields
    )
else:
    train, dev, test = datasets.SNLI.splits(inputs, answers)

    train_examples = [vars(x) for x in train]
    dev_examples = [vars(x) for x in dev]
    test_examples = [vars(x) for x in test]        
    
    with open('train.json', 'w+') as f:
        for example in train_examples:
            json.dump(example, f)
            f.write('\n')                            
    with open('test.json', 'w+') as f:
        for example in test_examples:
            json.dump(example, f)
            f.write('\n')
    with open('dev.json', 'w+') as f:
        for example in dev_examples:
            json.dump(example, f)
            f.write('\n')
            
inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(args.vector_cache))
        torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=128, device=device,
    sort_within_batch=False,
    sort_key=lambda x: 0)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

def compute_dev_acc(model):
    train_iter.init_epoch()
    test_iter.init_epoch()

    model.eval()

    # Compute test accuracy
    n_test_correct, test_loss = 0, 0
    k = 0
    with torch.no_grad():
        for test_batch_idx, test_batch in enumerate(test_iter):
            sys.stdout.flush()
            for i in range(len(test_batch)):
                sys.stdout.flush()
                b = data.Batch()
                b.premise = test_batch.premise[:,i].view(-1, 1)
                b.hypothesis = test_batch.hypothesis[:,i].view(-1,1)
                #answer = model(test_batch)
                #a1 = model(b)
                answer = model(b)
                pred = torch.max(answer, 1)[1].cpu().numpy().flatten()
                truth = test_batch.label[i].cpu().numpy().flatten()
                n_test_correct += np.sum(pred == truth)
                k += 1
                #print(n_test_correct / k)
                #n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()) == test_batch.label).sum().item()
    #print(n_test_correct)
    test_acc = 100. * n_test_correct / len(test)
    return test_acc

def compute_runtime(model):
    test_batch_idx, test_batch = next(enumerate(test_iter))
    test_batch.premise = test_batch.premise[:,1].view(-1, 1)
    test_batch.hypothesis = test_batch.hypothesis[:,1].view(-1, 1)    
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        tstart = time.time()
        for j in range(10):
            
            model(test_batch)
        torch.cuda.synchronize()
        tend = time.time()
        times.append(tend-tstart)
    tt = min(times)
    return tt
        

# Restore model

kwargs = {
    "method" : args.method,
    "W1_bits" : args.W_bits,
    "A1_bits" : args.A_bits,
    "W2_bits": args.W_bits,
    "A2_bits": args.A_bits,
}

model = SNLIClassifier(config, **kwargs).cuda()

# Load model
with open(args.model_path, "rb") as f:
    sd = torch.load(f).state_dict()
model.load_state_dict(sd)
model.eval()

acc = compute_dev_acc(model)
runtime = compute_runtime(model)

print("module,mnist,%s,%d,%d,%f,%f" % (args.method, args.W_bits, args.A_bits, acc, runtime))

