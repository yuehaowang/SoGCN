import argparse
import time, os
import pickle, json
import numpy as np

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from nets.ogb_proteins_nodeproppred.gnn import GCN, SAGE, GIN, \
    GCNII, APPNPNet, ARMANet, SoGCN

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from nets.ogb_proteins_nodeproppred.logger import Logger

def compute_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += int(np.prod(list(param.data.size())))
    return total_param

def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='sogcn',
                        help='GNN model to train (default: sogcn)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GNN message passing layers (default: 3)')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='dimensionality of hidden units in GNN (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout ratio (default: 0.0)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--eval_steps', type=int, default=5,
                        help='the epoch inverval to perform evaluation on test set (default: 5)')
    parser.add_argument('--log_steps', type=int, default=1,
                        help='the epoch interval to output the log information (default: 1)')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of runs to test results (default: 10)')
    parser.add_argument('--test', action='store_true',
                        help='set if user want to load logs from file to reproduce reults')
    parser.add_argument('--filename', type=str, default='',
                        help='filename to load or write output results (default: )')

    # additional parameter for SoGCN
    parser.add_argument('--K', type=int, default=2,
                        help='degree of polynomial filter. Only for SoGCN')

    # additional parameter for GCNII
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='hyperparameter alpha for GCNII/GCNII*')
    parser.add_argument('--theta', type=float, default=1.0,
                        help='hyperparameter theta for GCNII/GCNII*')

    # additional parameter for APPNP
    parser.add_argument('--teleport', type=float, default=0.1,
                        help='hyperparameter teleport (alpha) for APPNP')
    parser.add_argument('--mlp_layers', type=int, default=3,
                        help='layer number of MLP. Only for APPNP')

    # additional parameter for ARMA
    parser.add_argument('--num_stacks', type=int, default=3,
                        help='number of stacks (branches). Only for ARMA')
    parser.add_argument('--num_blocks', type=int, default=3,
                        help='number of blocks in each stack. Only for ARMA')

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins', root='data/OGB', transform=T.ToSparseTensor())
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.model == 'graphsage':
        model = SAGE(data.num_features, args.hidden_channels, 112,
                     args.num_layers, args.dropout).to(device)
    elif args.model == 'gcn':
        model = GCN(data.num_features, args.hidden_channels, 112,
                    args.num_layers, args.dropout).to(device)
    elif args.model == 'gin':
        model = GIN(data.num_features, args.hidden_channels, 112,
                    args.num_layers, args.dropout).to(device)
    elif args.model == 'gcnii':
        model = GCNII(data.num_features, args.hidden_channels, 112,
                      args.num_layers, args.dropout, alpha=args.alpha, theta=args.theta,
                      shared_weights=True).to(device)
    elif args.model == 'gcnii_star':
        model = GCNII(data.num_features, args.hidden_channels, 112,
                      args.num_layers, args.dropout, alpha=args.alpha, theta=args.theta,
                      shared_weights=False).to(device)
    elif args.model == 'appnp':
        model = APPNPNet(data.num_features, args.hidden_channels, 112,
                         args.num_layers, args.mlp_layers, args.dropout, alpha=args.teleport).to(device)
    elif args.model == 'arma':
        model = ARMANet(data.num_features, args.hidden_channels, 112,
                        args.num_layers, args.num_blocks, args.num_stacks, args.dropout).to(device)
    elif args.model == 'sogcn':
        model = SoGCN(data.num_features, args.hidden_channels, 112,
                      args.K, args.num_layers, args.dropout).to(device)

    MODEL_LIST_NORMALIZATION = ['gcn', 'gin', 'gcnii', 'gcnii_star', 'sogcn', 'appnp', 'arma']
    if args.model in MODEL_LIST_NORMALIZATION:
        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    print("Total parameters:", compute_model_param(model))

    ### RELOAD ONLY
    if args.test:
        if not os.path.exists(args.filename):
            print('Failed to load log file:', args.filename)
        else:
            print('Reloading results from log file:', args.filename)
            logger.load(args.filename)
            logger.print_statistics()
        return

    ### Start Training
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # reset save info
        train_acc = []
        val_acc = []
        test_acc = []
        train_time = []

        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss = train(model, data, train_idx, optimizer)
            train_time = time.time() - t0

            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result + (train_time,))

                train_rocauc, valid_rocauc, test_rocauc = result
                train_acc.append(train_rocauc)
                test_acc.append(valid_rocauc)
                test_acc.append(test_rocauc)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_rocauc:.2f}%, '
                        f'Valid: {100 * valid_rocauc:.2f}% '
                        f'Test: {100 * test_rocauc:.2f}% '
                        f'Time: {train_time:.2f}s')
        logger.print_statistics(run)
    logger.print_statistics()

    # save log files
    if args.filename != '':
        logger.save(args.filename)


if __name__ == "__main__":
    main()