import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import argparse
import os, time
import numpy as np

import pickle, json

from nets.ogb_molhiv_graphproppred.gnn import GNN

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def compute_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += int(np.prod(list(param.data.size())))
    return total_param

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def reload(filename):
    best_vals = []
    best_tests = []
    with open(filename, 'rb') as f:
        dict_runs = pickle.load(f)
        for _, data in dict_runs.items():
            train_curve = np.array(data['train'])
            val_curve = np.array(data['valid'])
            test_curve = np.array(data['test'])

            i_best = np.argmax(val_curve)
            best_vals.append(val_curve[i_best])
            best_tests.append(test_curve[i_best])

    best_vals = np.array(best_vals)
    print("Final validation score: {} ± {}".format(best_vals.mean(), best_vals.std()))
    best_tests = np.array(best_tests)
    print("Final test score: {} ± {}".format(best_tests.mean(), best_tests.std()))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='sogcn',
                        help='GNN model name (default: sogcn)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--runs', type=int, default=4,
                        help='number of runs to test results (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--test', action='store_true',
                        help='set if user want to load logs from file to reproduce reults')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to load or write output result (default: )')

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
    parser.add_argument('--stacks', type=int, default=3,
                        help='number of stacks (branches). Only for ARMA')
    parser.add_argument('--blocks', type=int, default=3,
                        help='number of blocks in each stack. Only for ARMA')

    args = parser.parse_args()
        
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset, root = 'data/OGB')

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    ### RELOAD ONLY
    if args.test:
        if not os.path.exists(args.filename):
            print('Failed to load log file:', args.filename)
        else:
            print('Reloading results from log file:', args.filename)
            reload(args.filename)
        return

    best_val, best_test = [], []
    train_time_all = []
    dict_save = {}

    for run in range(1, args.runs + 1):

        if args.gnn == 'gin':
            model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gin-virtual':
            model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        elif args.gnn == 'gcn':
            model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gcn-virtual':
            model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        elif args.gnn == 'sogcn':
            model = GNN(gnn_type = 'sogcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = False, K=args.K).to(device)
        elif args.gnn == 'gcnii':
            model = GNN(gnn_type = 'gcnii', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = False, alpha=args.alpha, theta=args.theta, shared_weights=True).to(device)
        elif args.gnn == 'gcnii_star':
            model = GNN(gnn_type = 'gcnii', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = False, alpha=args.alpha, theta=args.theta, shared_weights=False).to(device)
        elif args.gnn == 'graphsage':
            model = GNN(gnn_type = 'graphsage', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'appnp':
            model = GNN(gnn_type = 'appnp', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = False, alpha=args.teleport, mlp_layers=mlp_layers).to(device)
        elif args.gnn == 'arma':
            model = GNN(gnn_type = 'arma', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim,
                        drop_ratio = args.drop_ratio, virtual_node = False, num_stacks=args.stacks, num_blocks=args.blocks).to(device)
        else:
            raise ValueError('Invalid GNN type')

        # Print summary for the first run
        if run == 1:
            print("===Summary===")
            print("Type:", args.gnn)
            print("#Layers:", args.num_layer)
            print("#Dimension:", args.emb_dim)
            print("#Params:", compute_model_param(model))
            print("Runs:", args.runs)
            print("Epochs:", args.epochs)
            print("Log:", args.filename)
            print("=============")

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        valid_curve = []
        test_curve = []
        train_curve = []
        train_time = []

        for epoch in range(1, args.epochs + 1):
            print("=====Run {:02d} Epoch {}".format(run, epoch))
            print('Training...') # and timing
            t0 = time.time()
            train(model, device, train_loader, optimizer, dataset.task_type)
            train_time.append(time.time() - t0)

            print('Evaluating...')
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

        if 'classification' in dataset.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)

        # append to all the train time
        train_time_all += train_time

        print('Finished training!')

        train_time = np.array(train_time)
        print('Training time / epoch: {} ± {}'.format(train_time.mean(), train_time.std()))

        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        print('Test score: {}'.format(test_curve[best_val_epoch]))

        best_val.append(valid_curve[best_val_epoch])
        best_test.append(test_curve[best_val_epoch])

        dict_save['run_%d' % run] = dict(
            train = train_curve,
            valid = valid_curve,
            test = test_curve
        )

    # save to pickle
    if args.filename != '':
        with open(args.filename, 'wb') as f:
            pickle.dump(dict_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_time_all = np.array(train_time_all)
    print('Training time / epoch: {} ± {}'.format(train_time_all.mean(), train_time_all.std()))

    best_val = np.array(best_val)
    print('Final validation score: {} ± {}'.format(best_val.mean(), best_val.std()))
    best_test = np.array(best_test)
    print('Final Test score: {} ± {}'.format(best_test.mean(), best_test.std()))

if __name__ == "__main__":
    main()
