
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
    choices=models.model_creators.keys(),
    help='the model to evaluate learning curves on')
parser.add_argument('--normalizer', type=str, default=None, choices=('l2', 'tcn', None))
parser.add_argument('--freeze_embedder', type=bool, default=False)
parser.add_argument('--object_selection', type=bool, default=False)
parser.add_argument('--task', type=str, help='the relational games task')
parser.add_argument('--train_split', type=str, choices=('stripes', 'hexos', 'pentos'))
parser.add_argument('--test_split_size', type=int, default=5_000,
    help='size of sample from hold out sets to evaluate on')
parser.add_argument('--val_size', type=int, default=1_000)
parser.add_argument('--test_size', type=int, default=5_000)
parser.add_argument('--n_epochs', default=100, type=int, help='number of epochs to train each model for')
parser.add_argument('--batch_size', default=512, help='batch size')
parser.add_argument('--learning_rate', default=0.001, help='learning rate')
parser.add_argument('--early_stopping', default=False, type=bool, help='whether to use early stopping')
parser.add_argument('--train_size', default=-1, type=int, help='training set size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--start_trial', default=0, type=int, help='what to call first trial')
parser.add_argument('--wandb_project_name', default=None, type=str, help='W&B project name')
parser.add_argument('--seed', default=314159, help='random seed')
parser.add_argument('--ignore_gpu_assert', action='store_true')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# set random seed
tf.random.set_seed(args.seed)

# check if GPU is being used
print(tf.config.list_physical_devices())
if not args.ignore_gpu_assert:
    assert len(tf.config.list_physical_devices('GPU')) > 0

# set up W&B logging
import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

wandb_project_name = args.wandb_project_name
if wandb_project_name is None:
    wandb_project_name = f'relational_games-{args.task}'

#endregion

#region load data
utils.print_section('LOADING DATA')
data_path = '../../data/relational_games'

dataset_specs = np.load(f'{data_path}/dataset_specs.npy', allow_pickle=True).item()

task_datasets = data_utils.load_task_datasets(args.task, data_path, data_format='tfrecord', ds_specs=dataset_specs)

train_split_ds = task_datasets[args.train_split]
# train_split_ds = data_utils.load_ds_from_npz(filename)

train_ds, val_ds, test_ds = utils.split_ds(train_split_ds, val_size=args.val_size, test_size=args.val_size)
del train_split_ds

batch_size = args.batch_size
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

eval_datasets = {split: ds.take(args.test_split_size).batch(batch_size) for split, ds in task_datasets.items() if split != args.train_split}

#endregion

#region training setup
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
create_opt = lambda: tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
# metrics = [
#         tf.keras.metrics.BinaryAccuracy(name='acc'),
#         tf.keras.metrics.Precision(class_id=1, name='precision'),
#         tf.keras.metrics.Recall(class_id=1, name='recall'),
#         tf.keras.metrics.F1Score(average='weighted', name='f1'),
#         tf.keras.metrics.AUC(curve='ROC', multi_label=True, name='auc')
#         ]
metrics = ['acc']

def create_callbacks(data_size=None, batch_size=None):
    callbacks = [
        # TqdmCallback(data_size=data_size, batch_size=batch_size),
        wandb.keras.WandbMetricsLogger(log_freq='batch')
    ]

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, start_from_epoch=20))

    return callbacks

fit_kwargs = dict(epochs=args.n_epochs)
#endregion

# region train model
train_size = args.train_size

num_trials = args.num_trials # num of trials per train set size
start_trial = args.start_trial
# endregion

#region functions
def train_model(
    create_model, eval_model, fit_kwargs, create_callbacks,
    wandb_project_name, group_name,
    train_ds, val_ds, test_ds,
    train_size, start_trial, num_trials,
    ):

    for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
        run = wandb.init(project=wandb_project_name, group=group_name, name=f'trial = {trial}',
                        config={'trial': trial, 'group': group_name, 'train split': args.train_split})
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

#region train & evaluate model

if args.object_selection is not None:
    args.object_selection = models.get_obj_selection_by_task(args.task)

def create_model():
    model = models.model_creators[args.model](args.normalizer, args.freeze_embedder, args.object_selection)
    model.compile(loss=loss, optimizer=create_opt(), metrics=metrics) # compile
    model.build(input_shape=(None, *train_ds.element_spec[0].shape)) # build
    return model

group_name = models.get_group_name(args.model, args.normalizer, args.freeze_embedder, args.object_selection)

utils.print_section("TRAINING & EVALUATING MODEL")
train_model(
    create_model, eval_model, fit_kwargs, create_callbacks,
    wandb_project_name, group_name,
    train_ds, val_ds, test_ds,
    train_size, start_trial, num_trials
    )
#endregion