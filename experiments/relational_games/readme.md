# Relational Games Experiments

The relational games benchmark for relational reasoning was contributed by [Shanahan et. al.](https://arxiv.org/pdf/1905.10307.pdf) and the data is made available [here](https://console.cloud.google.com/storage/browser/relations-game-datasets;tab=objects?prefix=&forceOnObjectsSortingFiltering=false).

The benchmark consists of an array of binary classification tasks based on identifying relations between a grid of objects represented as abstract figures. A demo of the tasks can be found in `relational_games_task_demos.ipynb`.

`data_utils.py` contains utilities for loading the data. `serialize_data.py` contains utilities for serialize the `.npz` formatted data provided [here](https://console.cloud.google.com/storage/browser/relations-game-datasets;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) into the `.tfrecord` format. This enables the data to be streamed rather than needed to be loaded into memory all at once. `cnn_embedder.py` defines the architecture of the common CNN embedder, and `models.py` defines the architecture of each model we evaluate.

**Steps to reproduce results:**

1) Download the relational games dataset from [here](https://console.cloud.google.com/storage/browser/relations-game-datasets;tab=objects?prefix=&forceOnObjectsSortingFiltering=false).
2) Serialize the data by running
    ```
    python serialize_data.py --data_path {data_path} --all_npz_files
    ```
    where `{data_path}` is the location you downloaded the data into. `--all_npz_files` indicates that you'd like serialize all the `.npz` files in the given directory. You may also specify the files you'd like to serialize directly (e.g., if you'd like to evaluate on some of the tasks) by using the `--npz_files` argument.
3) For each `{model}` and `{task}` you'd like to evaluate, run the following code:
    ```
    python train_model.py --model {model} --task {task} --train_split pentos --n_epochs 50 --train_size -1 --start_trial 0 --num_trials 5
    ```
    This code evaluates the model on the given task and logs the results. In our experiments, we evaluate each `{model}` in `['relconvnet', 'corelnet', 'predinet', 'transformer']` and each `{task}` in `['same', 'occurs', 'xoccurs', '1task_between', '1task_match_patt']`.


The experimental logs from our runs are linked below.

| Task          	| Experimental logs                                         	|
|---------------	|-----------------------------------------------------------	|
| same          	| https://wandb.ai/relational-convolutions/relational_games-same             	|
| occurs        	| https://wandb.ai/relational-convolutions/relational_games-occurs           	|
| xoccurs       	| https://wandb.ai/relational-convolutions/relational_games-xoccurs          	|
| between       	| https://wandb.ai/relational-convolutions/relational_games-1task_between    	|
| match pattern 	| https://wandb.ai/relational-convolutions/relational_games-1task_match_patt 	|