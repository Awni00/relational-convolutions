import tensorflow as tf
import tensorflow_models as tfm
from cnn_embedder import CNNEmbedder

import sys; sys.path.append('..'); sys.path.append('../..')
from relational_neural_networks.multi_head_relation import MultiHeadRelation
from relational_neural_networks.relational_graphlet_convolution import RelationalGraphletConvolution


# RelConvNet
mhr_kwargs = dict(rel_dim=4, proj_dim=4, symmetric=True)
relconv_kwargs = dict(n_filters=8, graphlet_size=3,
        symmetric_inner_prod=True, groups_type='combinations',
        permutation_aggregator='max')
cnn_embedder_kwargs = dict(n_f=(16,16), s_f=(3,3), pool_size=2)

def create_relconvnet():
    mhr = MultiHeadRelation(**mhr_kwargs, name='mhr')
    rel_conv = RelationalGraphletConvolution(
        **relconv_kwargs, name='rgc')

    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    model = tf.keras.Sequential(
        [
            cnn_embedder,
            mhr,
            rel_conv,
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(2, name='output')],
        name='rel_conv_net')

    return model


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

    cnn_embedder = CNNEmbedder()

    model = tf.keras.Sequential([cnn_embedder, encoder, tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(2)])
    return model


## CoRelNet
