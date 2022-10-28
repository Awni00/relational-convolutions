import itertools
import numpy as np

import sklearn.model_selection
import imblearn.under_sampling

import tensorflow as tf


colors = (0, 1, 2)
shapes = (0, 1, 2)
fills = (0, 1, 2)
nums = (0, 1, 2)

color_dict = {0: 'red', 1: 'green', 2: 'blue'}
shape_dict = {0: 'bar', 1: 'squiggle', 2: 'diamond'}
fill_dict = {0: 'filled', 1: 'striped', 2: 'empty'}
num_dict = {0: '1', 1: '2', 2: '3'}

class_names = ['non-set', 'set']

entity_space = list(itertools.product(colors, shapes, fills, nums))

triplet_space = list(itertools.permutations(entity_space, 3))

# create a dictionary mapping each entity to a one_hot encoding of its attributes (e.g.: is_red, is_diamond, etc.)
entity_encodings = {}

for entity in entity_space:
    c, s, f, n = entity
    encoding = [0]*4*3
    encoding[c] = 1
    encoding[s + 3] = 1
    encoding[f + 6] = 1
    encoding[n + 9] = 1
    entity_encodings[entity] = tuple(encoding)

# define encoding function
def encode_entity(entity):
    return entity_encodings[entity]

def encode_triplet(triplet):
    return tuple(encode_entity(entity) for entity in triplet)


def classify_triplet(triplet):

    for attr in (0, 1, 2, 3): # check each attribute (color, shape, fill, num)
        attr_vals = [entity[attr] for entity in triplet]
        all_same = len(set(attr_vals)) == 1
        all_different = len(set(attr_vals)) == 3
        if not (all_same or all_different):
            return False

    return True

def create_Xy_data(encode=False):
    X = triplet_space
    y = np.array([classify_triplet(triplet) for triplet in triplet_space])

    if encode:
        X = np.array([encode_triplet(triplet) for triplet in X])
    else:
        X = np.array(X)

    return X, y

# samples of sets:
def get_entity_desc(entity):
    c, s, f, n = entity
    c = color_dict[c]
    s = shape_dict[s]
    f = fill_dict[f]
    n = num_dict[n]
    return (c, s, f, n)




def get_datasets(test_size=0.4, val_size=0.1):

    # get X, y data
    X, y = create_Xy_data(encode=True)

    # the data is highly imbalanced; create a balanced dataset by random undersampling
    resampler = imblearn.under_sampling.RandomUnderSampler()
    res_idx, y_res = resampler.fit_resample(np.array(range(len(y))).reshape(-1,1), y)
    X_enc_res = np.array(X) [np.squeeze(res_idx)]

    train_idx, test_idx = sklearn.model_selection.train_test_split(range(len(y_res)), test_size=test_size, stratify=y_res)
    X_test = [X_enc_res[i] for i in test_idx]
    y_test = [int(y_res[i]) for i in test_idx]

    y_train = [int(y_res[i]) for i in train_idx]

    train_idx, val_idx = sklearn.model_selection.train_test_split(train_idx, test_size=val_size/(1-test_size), stratify=y_train)

    X_train = [X_enc_res[i] for i in train_idx]
    y_train = [int(y_res[i]) for i in train_idx]
    X_val = [X_enc_res[i] for i in val_idx]
    y_val = [int(y_res[i]) for i in val_idx]

    y_train = tf.one_hot(y_train, 2)
    y_val = tf.one_hot(y_val, 2)
    y_test = tf.one_hot(y_test, 2)

    # create tensorflow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_ds, val_ds, test_ds