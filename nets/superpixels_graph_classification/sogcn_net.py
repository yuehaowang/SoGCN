import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl


from layers.sogcn_layer import SoGCNLayer
from layers.gru_layer import GRU
from layers.mlp_readout_layer import MLPReadout

activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity()
}


class SoGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        max_order = net_params['max_order']
        activation_name = net_params['activation']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        layer_params = {
            'activation': activations[activation_name],
            'dropout': dropout,
            'batch_norm': self.batch_norm,
            'residual': self.residual,
            'order': max_order
        }
        
        self.layers = nn.ModuleList([SoGCNLayer(hidden_dim, hidden_dim, **layer_params) for _ in range(n_layers-1)])
        self.layers.append(SoGCNLayer(hidden_dim, out_dim, **layer_params))
        self.gru = GRU(hidden_dim, hidden_dim) if net_params['gru'] else None
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h)
            if self.gru is not None and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t

        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

