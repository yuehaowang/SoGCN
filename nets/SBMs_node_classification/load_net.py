"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SBMs_node_classification.gcn_net import GCNNet
from nets.SBMs_node_classification.sogcn_net import SoGCNNet


def GCN(net_params):
    return GCNNet(net_params)

def SoGCN(net_params):
    return SoGCNNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'SoGCN': SoGCN
    }

    return models[MODEL_NAME](net_params)