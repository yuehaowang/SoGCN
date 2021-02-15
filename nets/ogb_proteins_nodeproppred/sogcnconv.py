from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch_scatter, torch_sparse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

# from torch_geometric.inits import glorot, zeros

def _sogcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, torch_sparse.SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.shape[1], ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = torch_scatter.scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class SoGCNConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    _cached_adj_t: Optional[torch_sparse.SparseTensor]

    def __init__(self, in_channels, out_channels, K=2, improved = False, cached = False,
                 add_self_loops = False, normalize = True, bias = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(SoGCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        # plus one for constant term
        self.weight = torch.nn.Parameter(torch.Tensor(K+1, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        torch.nn.init.xavier_uniform_(self.weight)
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

        out = torch.matmul(x, self.weight[0])
        for w in self.weight[1:]:
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            out += torch.matmul(x, w)

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
                                   self.out_channels, self.K)