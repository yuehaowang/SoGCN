"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import MAE

"""
    For GCNs
"""

def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].float().to(device)  # num x feat
        batch_targets = batch_targets.float().to(device)
        
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, None)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].float().to(device)
            batch_targets = batch_targets.float().to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, None)
            loss = model.loss(batch_scores, batch_targets)
            
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae
