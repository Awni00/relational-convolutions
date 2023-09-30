# SET Experiment

This experiment is based on the [SET game](https://www.wikiwand.com/en/Set_(card_game)). The objects in SET is cards with figures varying along four dimensions: color, shape, number, and fill. In SET, a player is given a sequence of cards, and must reason about the relations between groups of cards in order to find a 'set'---a triplet of cards satisfying a certain relational pattern.

The task in this experiment is: given a sequence of $k=5$ card images, determine whether or not it contains a 'set'. We compare RelConvNet to CoRelNet, PrediNet, and a Transformer.

`data_utils.py` and `setGame.py` contain utility functions for generating the dataset. `all-cards.png` contains images of each card in SET. `train_embedder.ipynb` builds the CNN embedder which is used in all models. `models.py` defines the model architectures. `train_model.py` evaluates a given model on the task and logs the results.

**Steps to reproduce results:**
1) Build the common CNN embedder by running `train_embedder.ipynb`. This trains a CNN on an auxiliary task, extracts an embedder from an intermediate layer, and saves the weights.
2) For each `{model}` in `['relconvnet', 'corelnet', 'predinet', 'transformer']`, run
    ```
    python train_model.py --model {model} --n_epochs 100 --train_size -1 --start_trial 0 --num_trials 10
    ```
    This evaluates the given model on the task and logs the results.

The experimental logs from our runs can be found here: https://wandb.ai/awni00/relconvnet-contains_set