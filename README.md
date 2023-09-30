# Relational Convolutional Networks

This is the project repo associated with the paper
> "Relational Convolutional Networks: A framework for learning representations of hierarchical relations" --- Awni Altabaa, John Lafferty

This repo includes an implementation of the proposed architecture as well as code for the experiments reported in the paper (see `experiments` directory for more details).

The following is an outline of the repo:

- `relational_neural_networks`: subdirectory containing implementation of 'relational models'.
    - `mdipr.py`: This implements the "Multi-Dimensional Inner Product Relation" layer.
    - `relational_graphlet_convolution.py`: This implements the "Relational Convolution" layer.
    - `grouping_layers.py`: This implements different grouping layers, including temporal grouping and feature-based grouping.
    - `predinet.py`: This implements the PrediNet model proposed [here](https://arxiv.org/pdf/1905.10307.pdf). PrediNet is one of the baselines we compare to (other baselines are CoRelNet and a Transformer).
    - `tcn.py`: This implements "context normalization". Some prior work which evaluated on the relational games benchmark used this as a preprocessing step. We chose not to use it since it is a confounder, but implemented it to examine its effects.

- `experiments`: subdirectory containing experiments involving relational models.
    - `relational_games`: This directory contains experiments on the relational games benchmark. See the readme for more details.
    - `set`: This directory contains experiments on the "contains set" benchmark. See the readme for more details.