# SoGCN: Second-Order Graph Convolutional Networks

- Based on [GNN Benchmark](https://github.com/graphdeeplearning/benchmarking-gnns)
- [Benchmark Installation](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/01_benchmark_installation.md)


# Reproduce Results

## GNN Benchmarks

### Test on Pretrained Models

1. Download our pretrained models.

    ```
    # make sure the commands below are executed in the root directory of this project
    bash scripts/download_pretrained_molecules.sh
    bash scripts/download_pretrained_superpixels.sh
    bash scripts/download_pretrained_SBMs.sh
    ```

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
    python main_SBMs_node_classification.py --model SoGCN --dataset SBM_PATTERN  --verbose True --gpu_id 0 --test True --out_dir _out/SBMs_node_classification/cluster_sogcn
    python main_SBMs_node_classification.py --model SoGCN --dataset SBM_PATTERN  --verbose True --gpu_id 0 --test True --out_dir _out/SBMs_node_classification/cluster_sogcn_gru

    ## PATTERN
    python main_SBMs_node_classification.py --model SoGCN --dataset SBM_PATTERN  --verbose True --gpu_id 0 --test True --out_dir _out/SBMs_node_classification/pattern_sogcn
    python main_SBMs_node_classification.py --model SoGCN --dataset SBM_PATTERN  --verbose True --gpu_id 0 --test True --out_dir _out/SBMs_node_classification/pattern_sogcn_gru
    ```



