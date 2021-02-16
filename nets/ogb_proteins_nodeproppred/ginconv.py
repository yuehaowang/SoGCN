import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

import math

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch_scatter, torch_sparse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from nets.ogb_proteins_nodeproppred.sogcnconv import _sogcn_norm


class GINConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    _cached_adj_t: Optional[torch_sparse.SparseTensor]

    def __init__(self, in_channels, out_channels, cached = False,
                 add_self_loops = False, normalize = True, bias = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GINConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        # plus one for constant term
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.eps = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        torch.nn.init.xavier_uniform_(self.weight)
        # glorot(self.eps)
        torch.nn.init.xavier_uniform_(self.eps)
        # zeros(self.bias)
        torch.nn.init.constant_(self.bias, 0.0)

        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_weight = None):

        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = _sogcn_norm(
                        edge_index, edge_weight, x.shape[self.node_dim],
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, torch_sparse.SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = _sogcn_norm(
                        edge_index, edge_weight, x.shape[self.node_dim],
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = torch.matmul(x, self.eps)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out += torch.matmul(x, self.weight)

        if self.bias is not None:
            out += self.bias

        return out

    # message passing scheme when edges are given
    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # take place when sparse tensor is given
    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
