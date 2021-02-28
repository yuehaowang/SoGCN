# SoGCN: Second-Order Graph Convolutional Networks

This is author's implementation of paper "SoGCN: Second-Order Graph Convolutional Networks" in PyTorch. All of the hyper-parameters and experiment settings have been included in this repo.

## Requirements

- Based on [GNN Benchmark](https://github.com/graphdeeplearning/benchmarking-gnns)
- [Benchmark Installation](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/01_benchmark_installation.md)

- Based on [Open Graph Benchmark](https://ogb.stanford.edu/)
- [Benchmark Installation](https://ogb.stanford.edu/docs/home/)

## Reproduce Results

For SGS and GNN benchmark datasets, we provide a script named 'scripts/exp.py' to run a series of model training in screen sessions. You can type `python scripts/exp.py -h` to view its usage. To OGB benchmark dataset, we provide shell scripts 'scripts/run_ogbn_proteins.sh' and 'scripts/run_ogbg_molhiv.sh' to reproduce results with our hyper-parameter settings.

For convenience, below presents the commands to reproduce our results.

## Synthetic Graph Spectrum Dataset

To train models on SGS datasets, run the following commands:
```
## High-Pass
python scripts/exp.py -a start -e highpass_sogcn -t SGS -g 1111 --dataset SGS_HIGH_PASS --config 'configs/SGS_node_regression_SoGCN.json'
python scripts/exp.py -a start -e highpass_sogcn -t SGS -g 1111 --dataset SGS_HIGH_PASS --config 'configs/SGS_node_regression_GCN.json'
python scripts/exp.py -a start -e highpass_sogcn -t SGS -g 1111 --dataset SGS_HIGH_PASS --config 'configs/SGS_node_regression_GIN.json'

## Low-Pass
python scripts/exp.py -a start -e lowpass_sogcn -t SGS -g 1111 --dataset SGS_LOW_PASS --config 'configs/SGS_node_regression_SoGCN.json'
python scripts/exp.py -a start -e lowpass_sogcn -t SGS -g 1111 --dataset SGS_LOW_PASS --config 'configs/SGS_node_regression_GCN.json'
python scripts/exp.py -a start -e lowpass_sogcn -t SGS -g 1111 --dataset SGS_LOW_PASS --config 'configs/SGS_node_regression_GIN.json'

## Band-Pass
python scripts/exp.py -a start -e bandpass_sogcn -t SGS -g 1111 --dataset SGS_BAND_PASS --config 'configs/SGS_node_regression_SoGCN.json'
python scripts/exp.py -a start -e bandpass_sogcn -t SGS -g 1111 --dataset SGS_BAND_PASS --config 'configs/SGS_node_regression_GCN.json'
python scripts/exp.py -a start -e bandpass_sogcn -t SGS -g 1111 --dataset SGS_BAND_PASS --config 'configs/SGS_node_regression_GIN.json'
```

Note the results will be saved to '_out/SGS_node_regression/'.

## Open Graph Benchmarks

Running the following commands will reproduce our results on Open Graph Benchmark datasets:
```
scripts/run_ogbn_proteins.sh <log_dir> [<gpu_id>] [--test]
scripts/run_ogbg_molhiv.sh <log_dir> [<gpu_id>] [--test]
```
where `log_dir` specifies the folder to load or save output logs. The downloaded log files will be saved in `_out/protein_nodeproppred` and `_out/molhiv_graphproppred` for `ogbn-protein` and `ogbn-molhiv` datasets, respectively. `gpu_id` specifies the GPU device to run our models. Add `--test` if you only want to reload the log files and read out the testing results. The OGB dataset will be automatically downloaded into `data/OGB` directory at the first run.

## GNN Benchmarks

To test on our pretrained models on GNN benchmarks, please follow the steps as below:

1. Download our pretrained models.

```
# make sure the commands below are executed in the root directory of this project
bash scripts/download_pretrained_molecules.sh
bash scripts/download_pretrained_superpixels.sh
bash scripts/download_pretrained_SBMs.sh
```

Pretrained models will be downloaded to '_out/molecules_graph_regression', '_out/superpixels_graph_classification', '_out/SBMs_node_classification', respectively.

2. Type the commands for different tasks

**Molecules Graph Regression**

```
## ZINC
python main_molecules_graph_regression.py --model SoGCN --dataset ZINC --gpu_id 0 --test True --out_dir _out/molecules_graph_regression/zinc_sogcn
python main_molecules_graph_regression.py --model SoGCN --dataset ZINC --gpu_id 0 --test True --out_dir _out/molecules_graph_regression/zinc_sogcn_gru
```

**Superpixels Graph Classification**

```
## MNIST
python main_superpixels_graph_classification.py --model SoGCN --dataset MNIST --gpu_id 0 --test True --out_dir _out/superpixels_graph_classification/mnist_sogcn
python main_superpixels_graph_classification.py --model SoGCN --dataset MNIST --gpu_id 0 --test True --out_dir _out/superpixels_graph_classification/mnist_sogcn_gru


## CIFAR10
python main_superpixels_graph_classification.py --model SoGCN --dataset CIFAR10 --gpu_id 0 --test True --out_dir _out/superpixels_graph_classification/cifar10_sogcn
python main_superpixels_graph_classification.py --model SoGCN --dataset CIFAR10 --gpu_id 0 --test True --out_dir _out/superpixels_graph_classification/cifar10_sogcn_gru
```

**SBMs Node Classification**

```
## CLUSTER
python main_SBMs_node_classification.py --model SoGCN --dataset SBM_CLUSTER  --verbose True --gpu_id 0 --test True --out_dir _out/SBMs_node_classification/cluster_sogcn
python main_SBMs_node_classification.py --model SoGCN --dataset SBM_CLUSTER  --verbose True --gpu_id 0 --test True --out_dir _out/SBMs_node_classification/cluster_sogcn_gru

## PATTERN
python main_SBMs_node_classification.py --model SoGCN --dataset SBM_PATTERN  --verbose True --gpu_id 0 --test True --out_dir _out/SBMs_node_classification/pattern_sogcn
python main_SBMs_node_classification.py --model SoGCN --dataset SBM_PATTERN  --verbose True --gpu_id 0 --test True --out_dir _out/SBMs_node_classification/pattern_sogcn_gru
```
