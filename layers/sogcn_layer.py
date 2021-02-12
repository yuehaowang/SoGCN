import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

import dgl.function as fn


# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}

# randwalk_msg = fn.copy_src(src='h', out='m')

# def randwalk_reduce(nodes):
#     accum = torch.mean(nodes.mailbox['m'], 1)
#     return {'h': accum}


def sym_normalized_msg(edges):
    return {'m': edges.src['h'] / (edges.src['out_deg'] * edges.dst['in_deg'])}

def sym_normalized_reduce(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}

'''
Formulation:
    A^2XW_2 + AXW_1 + IXW_0
'''

class SoGCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=False, order=2):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.max_order = order
        
        if in_dim != out_dim:
            self.residual = False

        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(order + 1)])

        self.bias = Parameter(torch.Tensor(out_dim))
        self._init_bias()
        
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def _init_bias(self):
        bound = 1 / math.sqrt(self.in_channels)
        init.uniform_(self.bias, -bound, bound)

    def _agg_readout_sum(self, outs):
        h = torch.stack(outs, dim=0).sum(dim=0)
        return h
    
    def forward(self, g, node_feat, intermediate=None):
        h_in = node_feat
        if intermediate is not None:
            intermediate(h_in, 'input')

        g = g.local_var()
        g.ndata['h'] = h_in
        g.ndata['in_deg'] = torch.sqrt(g.in_degrees().float()).unsqueeze(-1).to(h_in.device)  # Expand to N x 1
        g.ndata['out_deg'] = torch.sqrt(g.out_degrees().float()).unsqueeze(-1).to(h_in.device)  # Expand to N x 1

        outs = []
        for i in range(0, len(self.linears)):
            h_o = self.linears[i](g.ndata['h'])
            outs.append(h_o)
            g.update_all(sym_normalized_msg, sym_normalized_reduce)
        h = self._agg_readout_sum(outs)
        h += self.bias
        if intermediate is not None:
            intermediate(h, 'linear_filter')
        
        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization
        if intermediate is not None:
            intermediate(h, 'batch_norm')

        if self.activation:
            h = self.activation(h)  # non-linear activation
        if intermediate is not None:
            intermediate(h, 'activation')

        if self.residual:
            h = h_in + h  # residual connection
        if intermediate is not None:
            intermediate(h, 'residual_conn')
            
        h = self.dropout(h)
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={}, max_order={}, activation={})'.format(
            self.__class__.__name__,
            self.in_channels, self.out_channels, self.residual,
            self.max_order, self.activation)

