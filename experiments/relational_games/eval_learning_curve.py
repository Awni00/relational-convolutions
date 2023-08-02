
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from tqdm.keras import TqdmCallback
import argparse

import models
import data_utils

import sys; sys.path.append('../'); sys.path.append('../..')
import utils

#region setup
# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
    choices=('transformer', 'relconvnet', 'corelnet', 'abstractor'),
    help='the model to evaluate learning curves on')
parser.add_argument('--task', type=str, help='the relational games task')
parser.add_argument('--train_split', type=str, choices=('stripes', 'hexos', 'pentos'))
parser.add_argument('--test_split_size', type=int, default=5000,
    help='size of sample from hold out sets to evaluate on')
parser.add_argument('--n_epochs', default=500, type=int, help='number of epochs to train each model for')
parser.add_argument('--batch_size', default=32, help='batch size')
parser.add_argument('--early_stopping', default=True, type=bool, help='whether to use early stopping')
parser.add_argument('--min_train_size', default=500, type=int, help='minimum training set size')
parser.add_argument('--max_train_size', default=5000, type=int, help='maximum training set size')
parser.add_argument('--train_size_step', default=50, type=int, help='training set step size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--start_trial', default=0, type=int, help='what to call first trial')
parser.add_argument('--wandb_project_name',  type=str, help='W&B project name')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# check if GPU is being used
print(tf.config.list_physical_devices())
assert len(tf.config.list_physical_devices('GPU')) > 0

# set up W&B logging
import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

wandb_project_name = args.wandb_project_name

#endregion

#region load data
utils.print_section('LOADING DATA')
data_path = '../../data/relational_games'
# filename = f'{data_path}/1task_match_patt_pentos.npz'
# filename = f'{data_path}/1task_between_stripes.npz'

dataset_specs = np.load(f'{data_path}/dataset_specs.npy', allow_pickle=True).item()

task_datasets = data_utils.load_task_datasets(args.task, data_path, data_format='tfrecord', ds_specs=dataset_specs)

train_split_ds = task_datasets[args.train_split]
# train_split_ds = data_utils.load_ds_from_npz(filename)

train_ds, val_ds, test_ds = utils.split_ds(train_split_ds, val_size=0.1, test_size=0.2)
del train_split_ds

batch_size = 32
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

eval_datasets = {split: ds.take(args.test_split_size).batch(batch_size) for split, ds in task_datasets.items() if split != args.train_split}

#endregion

#region training setup
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
create_opt = lambda: tf.keras.optimizers.Adam()
metrics = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(class_id=1, name='precision'),
        tf.keras.metrics.Recall(class_id=1, name='recall'),
        tf.keras.metrics.F1Score(average='weighted', name='f1'),
        tf.keras.metrics.AUC(curve='ROC', multi_label=True, name='auc')
        ]

def create_callbacks(data_size=None, batch_size=None):
    callbacks = [
        # TqdmCallback(data_size=data_size, batch_size=batch_size),
        wandb.keras.WandbMetricsLogger(log_freq='epoch')
    ]

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, start_from_epoch=20))

    return callbacks

fit_kwargs = dict(epochs=args.n_epochs)
batch_size = 32
#endregion

# region evaluate learning curves
max_train_size = args.max_train_size
train_size_step = args.train_size_step
min_train_size = args.min_train_size
train_sizes = np.arange(min_train_size, max_train_size+1, step=train_size_step)

num_trials = args.num_trials # num of trials per train set size
start_trial = args.start_trial

print(f'will evaluate learning curve for `train_sizes` from {min_train_size} to {max_train_size} in increments of {train_size_step}.')
print(f'will run {num_trials} trials for each of the {len(train_sizes)} training set sizes for a total of {num_trials * len(train_sizes)} trials')
# endregion

#region functions
def evaluate_learning_curves(
    create_model, eval_model, fit_kwargs, create_callbacks,
    wandb_project_name, group_name,
    train_ds, val_ds, test_ds,
    train_sizes, start_trial, num_trials,
    ):

    for train_size in tqdm(train_sizes, desc='train size'):

        for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
            run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial, 'group': group_name, 'train split': args.train_split})
            model = create_model()

            train_ds_sample = train_ds.shuffle(buffer_size=len(train_ds)).take(train_size).batch(batch_size)
            history = model.fit(
                train_ds_sample, validation_data=val_ds, verbose=0,
                callbacks=create_callbacks(), **fit_kwargs)

            eval_dict = eval_model(model)
            wandb.log(eval_dict)
            wandb.finish(quiet=True)

            del model, train_ds_sample

def eval_model(model):
    eval_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    for split, ds in eval_datasets.items():
        ds_metrics = {f'{split}_{metric_name}': metric
            for metric_name, metric in model.evaluate(ds, return_dict=True, verbose=0).items()}
        eval_metrics = {**eval_metrics, **ds_metrics}

    return eval_metrics

# endregion

#region evaluate learning curves
create_model_dict = dict(
    relconvnet=models.create_relconvnet,
    transformer=models.create_transformer
    )

create_model = create_model_dict[args.model]
def create_model():
    model = create_model_dict[args.model]()
    model.compile(loss=loss, optimizer=create_opt(), metrics=metrics) # compile
    model.build(input_shape=(None, *train_ds.element_spec[0].shape)) # build
    return model

group_name = args.model

utils.print_section("EVALUATING LEARNING CURVES")
evaluate_learning_curves(
    create_model, eval_model, fit_kwargs, create_callbacks,
    wandb_project_name, group_name,
    train_ds, val_ds, test_ds,
    train_sizes, start_trial, num_trials
    )
#endregion