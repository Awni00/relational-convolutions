import itertools
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
from cnn_embedder import CNNEmbedder

import sys; sys.path.append('..'); sys.path.append('../..')
from relational_neural_networks.multi_head_relation import MultiHeadRelation
from relational_neural_networks.relational_graphlet_convolution import RelationalGraphletConvolution
from relational_neural_networks.tcn import TCN, GroupTCN
from relational_neural_networks.predinet import PrediNet

# global parameters
cnn_embedder_kwargs = dict(n_f=(16,16), s_f=(3,3), pool_size=2)
hidden_dense_size = 64

# RelConvNet
relconv_mhr_kwargs = dict(rel_dim=16, proj_dim=4, symmetric=True)
relconv_kwargs = dict(n_filters=16, graphlet_size=3,
        symmetric_inner_prod=False)

def create_relconvnet(normalizer=None, freeze_embedder=False, object_selection=None):
    object_selector = get_obj_selector(object_selection)
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    mhr1 = MultiHeadRelation(**relconv_mhr_kwargs, name='mhr1')
    rel_conv1 = RelationalGraphletConvolution(
        **relconv_kwargs, groups='combinations', name='rgc1')

    model = tf.keras.Sequential([
        object_selector,
        cnn_embedder,
        normalizer,
        mhr1,
        rel_conv1,
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1'),
        tf.keras.layers.Dense(2, name='output')
        ], name='relconv'
    )

    return model

def create_randomgroup_relconvnet(normalizer=None, freeze_embedder=False, object_selection=None):
    object_selector = get_obj_selector(object_selection) # NOTE: this is ignored for this model
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    mhr1 = MultiHeadRelation(**relconv_mhr_kwargs, name='mhr1')

    n_groups = 30
    groups = [tuple(group) for group in itertools.combinations(range(9), r=3)]
    groups_choice_idx = np.random.choice(len(groups), n_groups, replace=False)
    groups = [group for i,group in enumerate(groups) if i in groups_choice_idx]
    rel_conv1 = RelationalGraphletConvolution(
        **relconv_kwargs, groups=groups, name='rgc1')

    model = tf.keras.Sequential([
        cnn_embedder,
        normalizer,
        mhr1,
        rel_conv1,
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1'),
        tf.keras.layers.Dense(2, name='output')
        ], name='relconv'
    )

    return model

# CorelNet
corelnet_mhr_kwargs = dict(rel_dim=1, proj_dim=None, symmetric=True)

def create_corelnet(normalizer=None, freeze_embedder=False, object_selection=None):
    object_selector = get_obj_selector(object_selection)
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    corelnet = tf.keras.layers.Lambda(lambda x: tf.matmul(x, x, transpose_b=True), name='similarity_matrix')

    model = tf.keras.Sequential(
        [
            object_selector,
            cnn_embedder,
            normalizer,
            corelnet,
            tf.keras.layers.Softmax(axis=-1, name='softmax'),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1'),
            tf.keras.layers.Dense(2, name='output')],
        name='corelnet')
    return model

def create_nosoftmaxcorelnet(normalizer=None, freeze_embedder=False, object_selection=None):
    object_selector = get_obj_selector(object_selection)
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    corelnet = tf.keras.layers.Lambda(lambda x: tf.matmul(x, x, transpose_b=True), name='similarity_matrix')

    model = tf.keras.Sequential(
        [
            object_selector,
            cnn_embedder,
            normalizer,
            corelnet,
            # tf.keras.layers.Softmax(axis=-1, name='softmax'),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1'),
            tf.keras.layers.Dense(2, name='output')],
        name='corelnet')
    return model


predinet_kwargs = dict(key_dim=4, n_heads=4, n_relations=16, add_temp_tag=False)
def create_predinet(normalizer=None, freeze_embedder=False, object_selection=None):
    object_selector = get_obj_selector(object_selection)
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    model = tf.keras.Sequential(
        [
            object_selector,
            cnn_embedder,
            normalizer,
            PrediNet(**predinet_kwargs),
            tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1'),
            tf.keras.layers.Dense(2, name='output')],
        name='predinet')

    return model

## Transformer
encoder_kwargs = dict(num_layers=1, num_attention_heads=8, intermediate_size=32,
    activation='relu', dropout_rate=0.0, attention_dropout_rate=0.0,
    use_bias=False, norm_first=True, norm_epsilon=1e-06, intermediate_dropout=0.0)
def create_transformer(normalizer=None, freeze_embedder=False, object_selection=None):
    object_selector = get_obj_selector(object_selection)
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    encoder = tfm.nlp.models.TransformerEncoder(
        **encoder_kwargs)

    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    model = tf.keras.Sequential([
        object_selector,
        cnn_embedder,
        normalizer,
        encoder,
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2)])
    return model


# region utilities for model configurations
def get_normalizer(normalizer):
    if normalizer is None:
        return tf.keras.layers.Identity(name='identity')
    elif normalizer == 'l2':
        return tf.keras.layers.UnitNormalization(name='l2_norm')
    elif normalizer == 'tcn':
        return TCN(name='tcn')
    else:
        raise ValueError(f'unknown normalizer {normalizer}')

def get_obj_selector(object_selection):
    if object_selection is None:
        return tf.keras.layers.Identity(name='identity')
    else:
        return tf.keras.layers.Lambda(lambda x: tf.gather(x, object_selection, axis=1), name='object_selector')

def get_obj_selection_by_task(task):
    obj_selection_dict = {'1task_match_patt': [0,1,2,6,7,8]}
    if task in obj_selection_dict:
        return obj_selection_dict[task]
    else:
        print("task doesn't have prespecified object selection")
        return None

def get_group_name(model_name, normalizer=None, freeze_embedder=False, object_selection=None):
    group_name = model_name
    if normalizer is not None:
        group_name += f'-{normalizer}'
    if freeze_embedder:
        group_name += '-freeze_embedder'
    if object_selection is not None:
        group_name += '-w_obj_selection'
    return group_name
#

# put all model creators into a dictionary to interface with `eval_learning_curve.py`
model_creators = dict(
    relconvnet=create_relconvnet,
    randomgroup_relconvnet=create_randomgroup_relconvnet,
    transformer=create_transformer,
    corelnet=create_corelnet,
    nosoftmax_corelnet=create_nosoftmaxcorelnet,
    predinet=create_predinet,
    )