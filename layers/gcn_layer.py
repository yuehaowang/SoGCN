import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import GraphConv

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
    
# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
msg = fn.copy_src(src='h', out='m')
reduce = fn.mean(msg='m', out='h')

# def reduce(nodes):
#     accum = torch.mean(nodes.mailbox['m'], 1)
#     return {'h': accum}

class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}

class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=False, dgl_builtin=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        
        if in_dim != out_dim:
            self.residual = False
        
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if self.dgl_builtin == False:
            self.apply_mod = NodeApplyModule(in_dim, out_dim)
        else:
            self.conv = GraphConv(in_dim, out_dim)

        
    def forward(self, g, feature, intermediate=None):
        h_in = feature   # to be used for residual connection
        if intermediate is not None:
            intermediate(h_in, 'input')

        if self.dgl_builtin == False:
            g.ndata['h'] = feature
            g.update_all(msg, reduce)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h'] # result of graph convolution
        else:
            h = self.conv(g, feature)
        if intermediate is not None:
            intermediate(h, 'linear_filter')
        
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
        if intermediate is not None:
            intermediate(h, 'batch_norm')
       
        if self.activation:
            h = self.activation(h)
        if intermediate is not None:
            intermediate(h, 'activation')
        
        if self.residual:
            h = h_in + h # residual connection
        if intermediate is not None:
            intermediate(h, 'residual_conn')
            
        h = self.dropout(h)
        return h
    
    def __repr__(self):
        # return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
        #                                      self.in_channels,
        #                                      self.out_channels, self.residual)
        return '{}(in_channels={}, out_channels={}, residual={}, activation={})'.format(
            self.__class__.__name__,
            self.in_channels, self.out_channels, self.residual, self.activation)
