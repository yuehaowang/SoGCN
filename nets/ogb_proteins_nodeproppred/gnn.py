import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, APPNP, ARMAConv
from nets.ogb_proteins_nodeproppred.sogcnconv import SoGCNConv
from nets.ogb_proteins_nodeproppred.ginconv import GINConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GIN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(GINConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GCNII(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, alpha=0.5, theta=1.0, shared_weights=True):
        super(GCNII, self).__init__()

        self.conv_in = GCNConv(in_channels, hidden_channels, normalize=False)

        self.convs = torch.nn.ModuleList()
        for l in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer=l+1,
                              shared_weights=shared_weights, normalize=False))

        self.conv_out = GCNConv(hidden_channels, out_channels, normalize=False)

        self.dropout = dropout

    def reset_parameters(self):
        self.conv_in.reset_parameters()
        self.conv_out.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x = F.relu(self.conv_in(x, adj_t))
        x_0 = x
        for conv in self.convs:
            x = conv(x, x_0, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.convs[-1](x, x_0, adj_t))
        return self.conv_out(x, adj_t)

class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_iteration,
                 mlp_layers, dropout, alpha=0.1):
        super(APPNPNet, self).__init__()

        self.mlp = torch.nn.ModuleList()
        self.mlp.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(mlp_layers - 2):
            self.mlp.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.mlp.append(torch.nn.Linear(hidden_channels, out_channels))

        self.appnp = APPNP(num_iteration, alpha, dropout=dropout, normalize=False)

    def reset_parameters(self):
        self.appnp.reset_parameters()
        for linear in self.mlp:
            linear.reset_parameters()

    def forward(self, x, adj_t):
        for linear in self.mlp[:-1]:
            x = F.relu(linear(x))
        x = self.mlp[-1](x)
        return self.appnp(x, adj_t)

class ARMANet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 T, K, dropout):
        super(ARMANet, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(ARMAConv(in_channels, hidden_channels, num_stacks=K,
                                   num_layers=T, dropout=dropout, shared_weights=False))
        for l in range(num_layers-2):
            self.convs.append(ARMAConv(hidden_channels, hidden_channels, num_stacks=K,
                                       num_layers=T, dropout=dropout, shared_weights=False))
        self.convs.append(ARMAConv(hidden_channels, out_channels, num_stacks=K,
                                   num_layers=T, dropout=dropout, shared_weights=False))
                                   
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SoGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K,
                 num_layers, dropout):
        super(SoGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SoGCNConv(in_channels, hidden_channels, K, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(SoGCNConv(hidden_channels, hidden_channels, K, normalize=False))
        self.convs.append(SoGCNConv(hidden_channels, out_channels, K, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
