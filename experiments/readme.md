# Experiments

The code logs experimental results in real-time to the [`wandb.ai`](wandb.ai) experiment tracking tool. The experimental logs are publicly available. They include training and validation metrics tracked during training, test metrics after training, code/git state, resource utilization, etc. The links can be found in the `readme.md` file of each experiment.

The python environment used to run our experiments is in the `conda_environment.yml` file. You can replicate it via:
```
conda env create -f conda_environment.yml
```

Instructions for reproducing the experimental results of each experiment is in their respective subdirectories.