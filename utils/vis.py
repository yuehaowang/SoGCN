import torch
import random
import seaborn as sns
import numpy as np 
from torch.utils.data import DataLoader
from data.data import LoadData


def get_spectrum(h, U):
    s = torch.matmul(torch.transpose(U, 1, 0), h.type(torch.DoubleTensor))
    return torch.flip(s, (0,))


def plot_spectrum(x, ax, title=None, style='curves', x_label='Frequency', y_label='Magnitude', no_ticks=False, **kwargs):
    '''
    x: (input multi-channel signals, U)
    ax: axis
    title: subplot title
    style: curves | area
    x_label, y_label: labels for x-axis and y-axis
    no_ticks: no ticks ?
    '''
    
    s = get_spectrum(*x)
    
    if x_label:
        ax.set_xlabel(x_label, size=15)
    if y_label:
        ax.set_ylabel(y_label, size=15)
    if title:
        ax.set_title(title)
    
    if no_ticks:
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticklabels([])
    
    if style == 'curves':
        for i in range(s.shape[1]):
            si = s[:, i]
            si = si.detach().numpy()
            sns.lineplot(x=np.arange(0, si.shape[0]), y=si, ax=ax, **kwargs)
    elif style == 'area':
        min_s, _ = torch.min(s, dim=-1)
        min_s = min_s.detach().numpy()
        max_s, _ = torch.max(s, dim=-1)
        max_s = max_s.detach().numpy()
        avg_s = torch.mean(s, dim=-1)
        avg_s = avg_s.detach().numpy()
        
        ax.plot(avg_s)
        ax.fill_between(np.arange(min_s.shape[0]), y1=min_s, y2=max_s, color=(0.75, 0.75, 0.75))


class IntermediateRecorder:
    '''
    Used to record input/output of each layer
    '''

    def __init__(self, name=''):
        self.channels = {}
        self.target_name = name
    
    def __call__(self, x, channel='default'):
        if not channel in self.channels:
            self.channels[channel] = []

        self.channels[channel].append(x)
    
    def __getitem__(self, channel):
        if not channel in self.channels:
            return None
        
        return self.channels[channel]


class TestDataset:
    def __init__(self, dataset_name, dev=torch.device('cpu')):
        dataset = LoadData(dataset_name)
        test_loader = DataLoader(dataset.test, shuffle=False, drop_last=False, collate_fn=dataset.collate)
        self.test_data_ls = list(test_loader)
        self.device = dev

    def pick(self, idx=None):
        # If idx is None, randomly pick
        if idx == None:
            data_idx = random.randint(0, len(self.test_data_ls) - 1)
        else:
            data_idx = idx
        
        G, label = self.test_data_ls[data_idx]
        label = label.to(self.device)
        
        # Input
        X0 = G.ndata['feat'].to(self.device)
        E0 = G.edata['feat'].to(self.device) if 'feat' in G.edata else torch.zeros(0)
        
        # Eigen decomposition
        A = G.adjacency_matrix(transpose=True).to_dense().to(dtype=torch.float64)
        A[A > 0] = 1.0

        D_inv = torch.sum(A, dim=1)
        D_inv[torch.where(D_inv > 0)] = 1 / D_inv[np.where(D_inv > 0)]
        D_inv = torch.diag(D_inv)
        A_n = torch.sqrt(D_inv) @ A @ torch.sqrt(D_inv ** 0.5)

        Lam, U = torch.symeig(A_n, eigenvectors=True)
        U = U.type(torch.DoubleTensor)
        
        return (G, label, X0, E0, U)