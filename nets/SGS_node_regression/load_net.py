"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SGS_node_regression.gcn_net import GCNNet
from nets.SGS_node_regression.gin_net import GINNet
from nets.SGS_node_regression.sogcn_net import SoGCNNet

def GCN(net_params):
    return GCNNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def SoGCN(net_params):
    return SoGCNNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'GIN': GIN,
        'SoGCN': SoGCN
    }
        
    return models[MODEL_NAME](net_params)