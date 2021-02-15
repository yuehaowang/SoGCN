import pandas as pd
import torch
import numpy as np
import os
import json
from nets.molecules_graph_regression.load_net import gnn_model as molecules_gnn_model
from nets.SGS_node_regression.load_net import gnn_model as SGS_gnn_model
from nets.superpixels_graph_classification.load_net import gnn_model as superpixels_gnn_model
from nets.SBMs_node_classification.load_net import gnn_model as SBMs_gnn_model
from distutils.dir_util import copy_tree, remove_tree
from distutils.file_util import copy_file


def cp(dir_path, file_ls, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for item in ['results', 'logs', 'configs', 'checkpoints']:
        if not os.path.exists(os.path.join(target_path, item)):
            os.makedirs(os.path.join(target_path, item))

    
    for filename in file_ls:
        result_id = filename[7:-4]

        copy_file(os.path.join(dir_path, 'results', filename), os.path.join(target_path, 'results', filename))
        copy_file(os.path.join(dir_path, 'configs', 'config_' + result_id + '.txt'), os.path.join(target_path, 'configs', 'config_' + result_id + '.txt'))
        copy_tree(os.path.join(dir_path, 'logs', result_id), os.path.join(target_path, 'logs', result_id))
        copy_tree(os.path.join(dir_path, 'checkpoints', result_id), os.path.join(target_path, 'checkpoints', result_id))  

def rm_residual_files(dir_path):
    result_ids = [filename[7:-4] for filename in os.listdir(os.path.join(dir_path, 'results'))]
    
    rm_file_ls = []

    for item in ['logs', 'configs', 'checkpoints']:
        for filename in os.listdir(os.path.join(dir_path, item)):
            file_id = filename
            if item == 'configs':
                file_id = filename[7:-4]
            if not file_id in result_ids:
                file_path = os.path.join(dir_path, item, filename)
                
                if os.path.isdir(file_path):  
                    remove_tree(file_path)  
                elif os.path.isfile(file_path):
                    os.remove(file_path)
                
                rm_file_ls.append(file_path)
            
    return rm_file_ls

def _preprocess_json_str(s):
    return s.replace("'", '"').replace('True', 'true').replace('False', 'false')

def parse(dir_path):
    df = pd.DataFrame()

    dir_path = os.path.join(dir_path, 'results')

    colon_parse_table = {
        # {key_name}: [{end index}, {variable name}, {type}]
        'Dataset: ': [-1, 'dataset', str],
        'Model: ': [None, 'model', str],
        'Total Parameters: ': [None, '#params', str],
        'TEST MAE: ': [None, 'test_MAE', float],
        'TRAIN MAE: ': [None, 'train_MAE', float],
        'TEST ACCURACY: ': [None, 'test_ACC', float],
        'TRAIN ACCURACY: ': [None, 'train_ACC', float],
        'TEST F1: ': [None, 'test_F1', float],
        'TRAIN F1: ': [None, 'train_F1', float],
        'TEST HITS@10: ':  [None, 'test_hits@10', float],
        'TEST HITS@50: ':  [None, 'test_hits@50', float],
        'TEST HITS@100: ':  [None, 'test_hits@100', float],
        'TRAIN HITS@10: ':  [None, 'train_hits@10', float],
        'TRAIN HITS@50: ':  [None, 'train_hits@50', float],
        'TRAIN HITS@100: ':  [None, 'train_hits@100', float],
        'TEST ACCURACY averaged': [None, 'test_ACC_avg', float],
        'TRAIN ACCURACY averaged': [None, 'train_ACC_avg', float],
        'Convergence Time (Epochs): ': [None, 'convg_epochs', float],
        'Total Time Taken: ': [-4, 'total_time', float],
        'Average Time Per Epoch: ': [-2, 'avg_time_epoch', float]
    }

    for fn in os.listdir(dir_path):
        if not fn.endswith('.txt'):
            continue

        file_path = os.path.join(dir_path, fn)

        dataset = None
        model_name = None
        d = {}
        with open(file_path, 'r') as f:
            for ln in f:
                ln = ln.strip()

                for token in colon_parse_table:
                    conf = colon_parse_table[token]
                    if ln.startswith(token):
                        start = len(token)
                        end = conf[0] if conf[0] is not None else len(ln)
                        d[conf[1]] = conf[2](ln[start:end])
                
                if ln.startswith('params='):
                    params = json.loads(_preprocess_json_str(ln[len('params='):]))
                    d.update(params)
                
                if ln.startswith('net_params='):
                    net_params = json.loads(_preprocess_json_str(ln[len('net_params='):].replace('device(type=\'cuda\')', '"cuda"')))
                    d.update(net_params)

        d['filename'] = fn
        df = df.append(pd.DataFrame({k: [v] for k, v in d.items()}), ignore_index = True)

    return df.fillna(value={
        'max_order': 2,
        'gru': False,
        'undirected': False
    })


def _load_one_model(exp_dir, df, index, device, verbose, gnn_model):
    filename = df.iloc[index]['filename']
    dataset_name = df.iloc[index]['dataset']
    model_type = df.iloc[index]['model']
    model_name = os.path.basename(exp_dir)

    if verbose:
        print('\nModel #%s:' % index)
    
    model_seed = df.iloc[index]['seed']
    if verbose:
        print('Seed =', model_seed)
    
    ckpt_id = filename[len('results'):-len('.txt')]
    
    # Retrieve network params
    net_params = None
    with open(os.path.join(exp_dir, 'results', filename), 'r') as f:
        for ln in f:
            ln = ln.strip()

            if ln.startswith('net_params='):
                net_params = json.loads(_preprocess_json_str(ln[len('net_params='):].replace('device(type=\'cuda\')', '"cuda"')))
                del net_params['device']
                break
    if verbose:
        print('Net params =', net_params)
    
    # Search for the latest ckpt
    ckpt_dir = os.path.join(exp_dir, 'checkpoints', ckpt_id, 'RUN_')
    ckpt_fn = sorted(os.listdir(ckpt_dir))[-1]
    ckpt_path = os.path.join(ckpt_dir, ckpt_fn)
    if verbose:
        print('Ckpt path =', ckpt_path)
    
    # Restore ckpt
    if verbose:
        print('Restoring ckpt...')
    ckpt = torch.load(ckpt_path)
    model = gnn_model(model_type, net_params)
    model = model.to(device)
    model.load_state_dict(ckpt)
    model.eval()

    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    if verbose:
        print('Total parameters:', total_param)

    if verbose:
        print('Ckpt restored.\n')
    
    return {'model': model, 'dataset': dataset_name, 'model_name': model_name, 'path': ckpt_path, 'seed': model_seed, 'net_params': net_params}

def load_model(exp_dir, device=torch.device('cpu'), filter=None, only_best = True, verbose=True):
    exp_dir = exp_dir.strip()
    if exp_dir.endswith('/'):
        exp_dir = exp_dir[0:-1]
    
    df = parse(exp_dir)
    if filter:
        df = filter(df)
    if df.empty:
        raise Exception('No available model')

    print('== Details:')
    
    dataset_name = df.iloc[0]['dataset']
    if not df['dataset'].eq(dataset_name).all():
        raise Exception('Dataset not unique')
    print('Dataset =', dataset_name)

    model_type = df.iloc[0]['model']
    model_name = os.path.basename(exp_dir)
    if not df['model'].eq(model_type).all():
        raise Exception('Model type not unique')
    print('Model Type =', model_type, ', Model Name =', model_name)
    
    # Sort the ckpts by their performance
    if dataset_name in ['ZINC']:
        df = df.sort_values(by='test_MAE', ascending=True)
        gnn_model = molecules_gnn_model
    elif dataset_name in ['SGS_HIGH_PASS', 'SGS_LOW_PASS', 'SGS_BAND_PASS']:
        df = df.sort_values(by='test_MAE', ascending=True)
        gnn_model = SGS_gnn_model
    elif dataset_name in ['CIFAR10', 'MNIST']:
        df = df.sort_values(by='test_ACC', ascending=True)
        gnn_model = superpixels_gnn_model
    elif dataset_name in ['SBM_CLUSTER', 'SBM_PATTERN']:
        df = df.sort_values(by='test_ACC', ascending=True)
        gnn_model = SBMs_gnn_model

    if only_best:
        return _load_one_model(exp_dir, df, 0, device, verbose, gnn_model)
    else:
        res_ls = []
        for i in range(len(df)):
            res_ls.append(_load_one_model(exp_dir, df, i, device, verbose, gnn_model))
        
        print('Totally loaded %s %s models' % (len(res_ls), model_type))

        return res_ls
