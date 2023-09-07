import itertools
import tensorflow as tf
import tensorflow_models as tfm
from cnn_embedder import CNNEmbedder

import sys; sys.path.append('..'); sys.path.append('../..')
from relational_neural_networks.multi_head_relation import MultiHeadRelation
from relational_neural_networks.relational_graphlet_convolution import RelationalGraphletConvolution
from relational_neural_networks.tcn import TCN, GroupTCN

# global parameters
cnn_embedder_kwargs = dict(n_f=(16,16), s_f=(3,3), pool_size=2)

# RelConvNet
relconv_mhr_kwargs = dict(rel_dim=16, proj_dim=4, symmetric=True)
relconv_kwargs = dict(n_filters=16, graphlet_size=3,
        symmetric_inner_prod=False)
cnn_embedder_kwargs = dict(n_f=(16,16), s_f=(3,3), pool_size=2)

def create_relconvnet():
    mhr1 = MultiHeadRelation(**relconv_mhr_kwargs, name='mhr1')
    groups = [tuple(group) for group in itertools.combinations(range(9), r=3) if all(x in (0,1,2,6,7,8) for x in group)]
    rel_conv1 = RelationalGraphletConvolution(
        **relconv_kwargs, groups=groups, name='rgc1')
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    l2_normalizer = tf.keras.layers.UnitNormalization(name='l2_normalization')

    model = tf.keras.Sequential([
        cnn_embedder,
        l2_normalizer,
        mhr1,
        rel_conv1,
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(64, activation='relu', name='hidden_dense1'),
        tf.keras.layers.Dense(2, name='output')
        ], name='relconv'
    )

    return model

# CorelNet
corelnet_mhr_kwargs = dict(rel_dim=1, proj_dim=None, symmetric=True)
def create_corelnet():
    mhr = MultiHeadRelation(**corelnet_mhr_kwargs, name='mhr')
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    model = tf.keras.Sequential(
        [
            cnn_embedder,
            tf.keras.layers.UnitNormalization(),
            mhr,
            tf.keras.layers.Softmax(axis=-1, name='softmax'),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(32, activation='relu', name='hidden_dense1'),
            tf.keras.layers.Dense(2, name='output')],
        name='corelnet')

    return model

def create_nosoftmaxcorelnet():
    mhr = MultiHeadRelation(**corelnet_mhr_kwargs, name='mhr')
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    model = tf.keras.Sequential(
        [
            cnn_embedder,
            tf.keras.layers.UnitNormalization(),
            mhr,
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(32, activation='relu', name='hidden_dense1'),
            tf.keras.layers.Dense(2, name='output')],
        name='nosoftmax_corelnet')

    return model


def create_tcncorelnet():
    mhr = MultiHeadRelation(**corelnet_mhr_kwargs, name='mhr')
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    model = tf.keras.Sequential(
        [
        cnn_embedder,
        TCN(), # TCN? axis=1 is the temporal axis
        mhr,
        tf.keras.layers.Softmax(axis=-1, name='softmax'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(32, activation='relu', name='hidden_dense1'),
        tf.keras.layers.Dense(2, name='output')],
        name='tcn_corelnet')

    return model

def create_grouptcn_corelnet():
    mhr = MultiHeadRelation(**corelnet_mhr_kwargs, name='mhr')
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    model = tf.keras.Sequential(
        [
            cnn_embedder,
            GroupTCN(groups=[(0,1,2), (6,7,8)]), # this encodes additional information about the problem
            mhr,
            tf.keras.layers.Softmax(axis=-1, name='softmax'),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(32, activation='relu', name='hidden_dense1'),
            tf.keras.layers.Dense(2, name='output')],
        name='corelnet')

    return model

predinet_kwargs = dict(key_dim=4, n_heads=4, n_relations=16, add_temp_tag=False)
def create_predinet():
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    model = tf.keras.Sequential(
        [
            cnn_embedder,
            PrediNet(**predinet_kwargs),
            tf.keras.layers.Dense(64, activation='relu', name='hidden_dense1'),
            tf.keras.layers.Dense(2, name='output')],
        name='predinet')

    return modedl

## Transformer
encoder_kwargs = dict(num_layers=1,
        num_attention_heads=8,
        intermediate_size=32,
        activation='relu',
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-06,
        intermediate_dropout=0.0)
def create_transformer():
    encoder = tfm.nlp.models.TransformerEncoder(
        **encoder_kwargs)

    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    model = tf.keras.Sequential([cnn_embedder, encoder, tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(2)])
    return model



# put all model creators into a dictionary to interface with `eval_learning_curve.py`
model_creators = dict(
    relconvnet=create_relconvnet,
    transformer=create_transformer,
    corelnet=create_corelnet,
    nosoftmax_corelnet=create_nosoftmaxcorelnet,
    tcn_corelnet=create_tcncorelnet,
    grouptcn_corelnet=create_grouptcn_corelnet,
    predinet=create_predinet,
    )
