import torch
import pickle
import time
import os
import math
import numpy as np

import dgl

import numpy as np
import scipy.stats as STATS


class SGSSplit(torch.utils.data.Dataset):
    def __init__(self, data, split):
        self.graph_lists = data[0]
        self.graph_labels = data[1]
        self.split = split
    
    def __len__(self):
        return len(self.graph_lists)
    
    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


class SGSDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SGS datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = './data/SGS/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = SGSSplit(f[0], 'train')
            self.val = SGSSplit(f[1], 'val')
            self.test = SGSSplit(f[2], 'test')

            indegrees = [torch.max(g.in_degrees()) for g in self.train.graph_lists + self.val.graph_lists + self.test.graph_lists]
            outdegrees = [torch.max(g.out_degrees()) for g in self.train.graph_lists + self.val.graph_lists + self.test.graph_lists]
            print('max in-degrees, out-degrees:', max(indegrees).item(), max(outdegrees).item())

        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels)
        
        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels
        
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]
