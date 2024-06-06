# Relational Convolutional Networks

This is the project repo associated with the paper
> "Learning Hierarchical Relational Representations through Relational Convolutions" --- Awni Altabaa, John Lafferty

The arXiv version is available here: https://arxiv.org/abs/2310.03240. The project webpage contains a high-level summary of the paper and can be found here: https://awni00.github.io/relational-convolutions.

This repo includes an implementation of the proposed architecture as well as code for the experiments reported in the paper (see `experiments` directory for more details).

The following is an outline of the repo:

- `relational_neural_networks`: subdirectory containing implementations of 'relational modules'.
    - `mdipr.py`: This implements the "Multi-Dimensional Inner Product Relation" layer.
    - `relational_graphlet_convolution.py`: This implements the "Relational Convolution" layer.

- `experiments`: subdirectory containing experiments involving relational models.
    - `relational_games`: This directory contains experiments on the relational games benchmark. See the readme for more details.
    - `set`: This directory contains experiments on the "contains set" benchmark. See the readme for more details.