"""
    File to load dataset based on user control from main file
"""
from data.superpixels import SuperPixDataset
from data.molecules import MoleculeDataset
from data.SBMs import SBMsDataset
from data.SGS import SGSDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)
    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDataset(DATASET_NAME)

    # handling for the SGS (Synthetic Graph Spectrum) Dataset
    SGS_DATASETS = ['SGS_HIGH_PASS', 'SGS_BAND_PASS', 'SGS_LOW_PASS']
    if DATASET_NAME in SGS_DATASETS: 
        return SGSDataset(DATASET_NAME)
    