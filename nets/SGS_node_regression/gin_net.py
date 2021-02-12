import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP

activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity()
}

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type    
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        activation_name = net_params['activation']
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        for layer in range(self.n_layers - 1):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim, activations[activation_name])
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, batch_norm, residual, 0, learn_eps, activations[activation_name]))

        self.ginlayers.append(GINLayer(ApplyNodeFunc(MLP(n_mlp_layers, hidden_dim, hidden_dim, out_dim, activations[activation_name])), neighbor_aggr_type,
                                dropout, batch_norm, residual, 0, learn_eps, activations[activation_name]))
        
    def forward(self, g, h, e, intermediate=None):
        h = self.embedding_h(h)

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, intermediate)

            if intermediate is not None:
                intermediate(h, 'out')

        return h
        
    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss
