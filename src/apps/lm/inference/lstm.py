import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lstm
from lstm import *
from qlinear import QLinear

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=.5, W_bits=32, A_bits=32, method="fake", batch_first=True):
        """Initialize params."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        self.input_weights = QLinear(input_size, 4 * hidden_size, W_bits=W_bits, A_bits=A_bits, method=method, optimize_cutoff=False)
        self.hidden_weights = QLinear(hidden_size, 4 * hidden_size, W_bits=W_bits, A_bits=A_bits, method=method, optimize_cutoff=False)

    def forward(self, input, hidden):
        """Propogate input through the network."""
        # tag = None  #
        def recurrence(input, hidden):
            """Recurrence helper."""

            input = self.dropout(input)

            hidden = [x.view(1, -1, self.hidden_size) for x in hidden]
            
            hx, cx = hidden  # n_b x hidden_dim

            hx = hx.reshape((1, -1))

            gates = self.input_weights(input) + \
                    self.hidden_weights(hx)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, -1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)  # o_t
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return self.dropout(output.view(-1, self.hidden_size)), hidden
