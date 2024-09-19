# Relational Convolutional Networks

<p align='center'>
    <a href=https://arxiv.org/abs/2310.03240>
        <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-paper-brown?logo=arxiv">
    </a>
    <a href=https://awni.xyz/relational-convolutions>
        <img alt="Project Webpage" src="https://img.shields.io/badge/www-website-royalblue">
    </a>
    <a href=https://wandb.ai/relational-convolutions/projects>
        <img alt="Experimental Logs" src="https://img.shields.io/badge/W&B-experiment logs-yellow">
    </a>
</p>

This is the project repo associated with the paper *[Learning Hierarchical Relational Representations through Relational Convolutions](https://arxiv.org/abs/2310.03240)* by Awni Altabaa, John Lafferty.

> **Abstract.** An evolving area of research in deep learning is the study of architectures and inductive biases that support the learning of relational feature representations. In this paper, we address the challenge of learning representations of *hierarchical* relations—that is, higher-order relational patterns among groups of objects. We introduce “relational convolutional networks”, a neural architecture equipped with computational mechanisms that capture progressively more complex relational features through the composition of simple modules. A key component of this framework is a novel operation that captures relational patterns in groups of objects by convolving graphlet filters—learnable templates of relational patterns—against subsets of the input. Composing relational convolutions gives rise to a deep architecture that learns representations of higher-order, hierarchical relations. We present the motivation and details of the architecture, together with a set of experiments to demonstrate how relational convolutional networks can provide an effective framework for modeling relational tasks that have hierarchical structure.

See the project webpage for a high-level summary of the key ideas presented in the paper: https://awni00.github.io/relational-convolutions.


## Outline of Codebase

This repo includes an implementation of the proposed architecture as well as code and instructions for reproducing the experiments reported in the paper (see `experiments` directory for more details).

The following is an outline of the repo:

- `relational_neural_networks`: subdirectory containing implementations of 'relational modules'.
    - `mdipr.py`: This implements the "Multi-Dimensional Inner Product Relation" layer.
    - `relational_graphlet_convolution.py`: This implements the "Relational Convolution" layer.

- `experiments`: subdirectory containing experiments involving relational models.
    - `relational_games`: This directory contains experiments on the relational games benchmark. See the readme for more details.
    - `set`: This directory contains experiments on the "contains set" benchmark. See the readme for more details.

## Usage

```python
from relational_neural_networks.relational_graphlet_convolution import RelationalGraphletConvolutionGroupAttn

relconv = RelationalGraphletConvolutionGroupAttn(
    n_filters=16,        # number of graphlet filters
    graphlet_size=3,     # size of graphlet filter (i.e., num of objects)
    n_groups=16,         # number of groups to learn via Group Attention
    mdipr_kwargs=dict(
        rel_dim=16,      # relation dimension (i.e., num relations)
        proj_dim=None,   # projection dim for computing each relation
        symmetric=False  # whether relations are symmetric
        ),
    group_attn_key_dim=16, # key dimension for group attention
    group_attn_key='pos+feat', # information used to determine groups
    symmetric_inner_prod=False, # whether relational inner product should be permutation-invariant
    permutation_aggregator='mean', # if symmetric_inner_prod=True, how to pool over permutations
    filter_initializer='random_normal', # how to initialize graphlet filters
    entropy_reg=False,     # whether to use entropy regularization on group attention scores
    entropy_reg_scale=1.0  # scaling factor for entropy regularization term (added to loss)
    )

relconv(tf.random.random(shape=(batch_size, n_objects, obj_dim)))
# output shape: (batch_size, n_groups, n_filters)

```

## Citation

If you use natbib or bibtex please use the following citation (as provided by Google Scholar).
```bibtex
@article{altabaa2024learninghierarchicalrelationalrepresentations,
    title={Learning Hierarchical Relational Representations through Relational Convolutions}, 
    author={Awni Altabaa and John Lafferty},
    year={2024},
    journal={arXiv preprint arXiv:2310.03240}
}
```

If you use `biblatex`, please use the following citation (as provided by arxiv).
```bibtex
@misc{altabaa2024learninghierarchicalrelationalrepresentations,
      title={Learning Hierarchical Relational Representations through Relational Convolutions}, 
      author={Awni Altabaa and John Lafferty},
      year={2024},
      eprint={2310.03240},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.03240}, 
}
```