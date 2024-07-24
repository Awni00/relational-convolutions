
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from tqdm.keras import TqdmCallback
import argparse

import models
from setGame import SetGame
import data_utils
from data_utils import sample_set, sample_nonset
from einops import rearrange

import sys; sys.path.append('../'); sys.path.append('../..')
import utils

#region setup
# parse script arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str,
#     choices=models.model_creators.keys(),
#     help='the model to evaluate learning curves on')
parser.add_argument('--val_size', type=int, default=0.15)
parser.add_argument('--test_size', type=int, default=0.15)
parser.add_argument('--n_epochs', default=100, type=int, help='number of epochs to train each model for')
parser.add_argument('--batch_size', default=512, help='batch size')
parser.add_argument('--learning_rate', default=0.001, help='learning rate')
parser.add_argument('--early_stopping', default=False, type=bool, help='whether to use early stopping')
parser.add_argument('--train_size', default=-1, type=int, help='training set size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--start_trial', default=0, type=int, help='what to call first trial')
parser.add_argument('--wandb_project_name', default='relconvnet-contains_set', type=str, help='W&B project name')
parser.add_argument('--seed', default=None, help='random seed')
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
#endregion

#region load data
utils.print_section('LOADING DATA')

val_size=args.val_size
test_size=args.test_size

# get all possible triples and parition into SETs and non-SETs
# split set triples into train/val/test
setgame = SetGame()
_, set_labels, triples = setgame.generate_data()

set_triples = [triple for triple, is_set in zip(triples, set_labels) if is_set]

def create_set_classification_dataset(num_seqs, k, set_triples):

    vocab_size = 81
    setgame = SetGame()

    # get embedding for each card
    card_shape = setgame.image_of_card(0,0).shape
    card_images = np.zeros((9, 9, *card_shape))
    for i in range(9):
        for j in range(9):
            card_images[i,j] = setgame.image_of_card(i, j)

    object_seqs = np.zeros((num_seqs, k, *card_shape))
    card_seqs = np.zeros((num_seqs, k, 2), dtype=int)
    labels = np.zeros(num_seqs, dtype=int)

    # sample tuples containing sets
    set_tuples = [sample_set(k, set_triples) for _ in range(num_seqs//2)]
    nonset_tuples = [sample_nonset(k) for _ in range(num_seqs//2)]

    # sample tuples not containing set

    # get card image embedding for each and create object_seqs, card_seqs, etc

    for s in np.arange(0, num_seqs, 2):
        for i in np.arange(k):
            card = set_tuples[s//2][i]
            object_seqs[s, i] = card_images[card[0], card[1]]
            card_seqs[s, i] = [card[0], card[1]]
        labels[s] = 1
        for i in np.arange(k):
            card = nonset_tuples[s//2][i]
            object_seqs[s+1, i] = card_images[card[0], card[1]]
            card_seqs[s+1, i] = [card[0], card[1]]
        labels[s+1] = 0

    object_seqs = rearrange(object_seqs, 'b k h w c -> b h (k w) c')

    return card_images, card_seqs, labels, object_seqs

# def train_val_test_split(X, val_size, test_size):
#     from sklearn.model_selection import train_test_split
#     X_train, X_test = train_test_split(X, test_size=test_size)
#     X_train, X_val = train_test_split(X_train, test_size=val_size/(1-test_size))
#     return X_train, X_val, X_test

# set_triples_train, set_triples_val, set_triples_test = train_val_test_split(set_triples, val_size=val_size, test_size=test_size)

# print(f'train SETs: {len(set_triples_train)}; val SETs: {len(set_triples_val)}; test SETs: {len(set_triples_test)}')

# k = 5 # length of card sequence in wich to determine if a SET exists
# # card_embedder = tf.keras.models.load_model('cnn_card_embedder/embedder')
# _, _, labels_train, object_seqs_train = create_set_classification_dataset(
#     num_seqs=len(set_triples_train)*24, k=k, set_triples=set_triples_train)
# _, _, labels_val, object_seqs_val = create_set_classification_dataset(
#     num_seqs=len(set_triples_val)*24, k=k, set_triples=set_triples_val)
# _, _, labels_test, object_seqs_test = create_set_classification_dataset(
#     num_seqs=len(set_triples_test)*24, k=k, set_triples=set_triples_test)

# X_train, X_val, X_test = object_seqs_train, object_seqs_val, object_seqs_test
# y_train, y_val, y_test = labels_train, labels_val, labels_test

# batch_size = args.batch_size

# train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# seq_len, dim = train_ds.element_spec[0].shape

val_size = args.val_size
test_size = args.test_size

def train_val_test_split(X, val_size, test_size):
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=test_size)
    X_train, X_val = train_test_split(X_train, test_size=val_size/(1-test_size))
    return X_train, X_val, X_test

set_triples_train, set_triples_val, set_triples_test = train_val_test_split(set_triples, val_size=val_size, test_size=test_size)

print(f'train SETs: {len(set_triples_train)}; val SETs: {len(set_triples_val)}; test SETs: {len(set_triples_test)}')


k = 5 # length of card sequence in wich to determine if a SET exists
_, _, labels_train, object_seqs_train = create_set_classification_dataset(
    num_seqs=len(set_triples_train), k=k, set_triples=set_triples_train)
_, _, labels_val, object_seqs_val = create_set_classification_dataset(
    num_seqs=len(set_triples_val), k=k, set_triples=set_triples_val)
_, _, labels_test, object_seqs_test = create_set_classification_dataset(
    num_seqs=len(set_triples_test), k=k, set_triples=set_triples_test)

X_train, X_val, X_test = object_seqs_train, object_seqs_val, object_seqs_test
y_train, y_val, y_test = labels_train, labels_val, labels_test

batch_size = args.batch_size
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

#endregion

#region training setup
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
create_opt = lambda: tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
metrics = ['acc']

def create_callbacks(data_size=None, batch_size=None):
    callbacks = [
        wandb.keras.WandbMetricsLogger(log_freq='epoch')
    ]

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, start_from_epoch=20))

    return callbacks

fit_kwargs = dict(epochs=args.n_epochs)
#endregion

# region train model
train_size = args.train_size

num_trials = args.num_trials # num of trials
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
        run = wandb.init(project=wandb_project_name, group=group_name, name=f'{group_name} (trial = {trial})',
                        config={'trial': trial, 'group': group_name})
        model = create_model()

        train_ds_sample = train_ds.shuffle(buffer_size=len(train_ds)).take(train_size).batch(batch_size)
        history = model.fit(
            train_ds_sample, validation_data=val_ds, verbose=1,
            callbacks=create_callbacks(), **fit_kwargs)

        eval_dict = eval_model(model)
        wandb.log(eval_dict)
        wandb.finish(quiet=True)

        del model, train_ds_sample

def eval_model(model):
    eval_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)

    return eval_metrics

# endregion

#region train & evaluate model

def create_cnn_model():
    n_f = [16, 16, 32, 32, 64, 64, 128, 128, 128, 128]
    s_f = [3]*10
    pool_sizes = [2, None, 2, None, 2, None, 2, (1,2), (1,2), 2]

    layers = []
    assert len(n_f) == len(s_f) == len(pool_sizes)

    for l in range(len(n_f)):
        layers.append(tf.keras.layers.Conv2D(n_f[l], s_f[l], activation='relu', padding='same'))
        if pool_sizes[l] is not None:
            layers.append(tf.keras.layers.MaxPool2D(pool_sizes[l]))

    layers += [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu', name='hidden_dense1'), # NOTE: changed from hidden_dense_size = 64
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, name='output')
    ]

    return tf.keras.Sequential(layers)

#region train & evaluate model
def create_model():
    model = create_cnn_model()
    model.compile(loss=loss, optimizer=create_opt(), metrics=metrics) # compile
    model.build(input_shape=(None, *train_ds.element_spec[0].shape)) # build
    return model

group_name = 'cnn' # args.model

utils.print_section("TRAINING & EVALUATING MODEL")
train_model(
    create_model, eval_model, fit_kwargs, create_callbacks,
    wandb_project_name, group_name,
    train_ds, val_ds, test_ds,
    train_size, start_trial, num_trials
    )
#endregion