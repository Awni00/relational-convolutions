import tensorflow as tf
import tensorflow_models as tfm

import sys; sys.path.append('..'); sys.path.append('../..')
from relational_neural_networks.mdipr import MultiDimInnerProdRelation
from relational_neural_networks.relational_graphlet_convolution import RelationalGraphletConvolution
from relational_neural_networks.predinet import PrediNet
from misc.abstractor import RelationalAbstracter

def create_predictormlp():
    predictor_mlp = tf.keras.Sequential([
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(64, activation='relu', name='hidden_dense1'),
        tf.keras.layers.Dense(32, activation='relu', name='hidden_dense2'),
        tf.keras.layers.Dense(2, name='output')
    ])
    return predictor_mlp

# region RelConvNet

relconv_mhr_kwargs = dict(rel_dim=16, proj_dim=16, symmetric=True)
relconv_kwargs = dict(n_filters=16, graphlet_size=3,
        symmetric_inner_prod=True, permutation_aggregator='max')
def create_relconvnet():
    mhr1 = MultiDimInnerProdRelation(**relconv_mhr_kwargs, name='mhr1')
    rel_conv1 = RelationalGraphletConvolution(
        **relconv_kwargs, groups='combinations', name='rgc1')

    model = tf.keras.Sequential([
        mhr1,
        rel_conv1,
        create_predictormlp()
        ], name='relconv'
    )

    return model

def create_relconvnet_maxpooling():
    relconv_mhr_kwargs = dict(rel_dim=16, proj_dim=16, symmetric=True)
    relconv_kwargs = dict(n_filters=16, graphlet_size=3,
            symmetric_inner_prod=True, permutation_aggregator='max')
    mhr1 = MultiDimInnerProdRelation(**relconv_mhr_kwargs, name='mhr1')
    rel_conv1 = RelationalGraphletConvolution(
        **relconv_kwargs, groups='combinations', name='rgc1')

    model = tf.keras.Sequential([
        mhr1,
        rel_conv1,
        tf.keras.layers.GlobalMaxPooling1D(),
        create_predictormlp()
        ], name='relconv'
    )

    return model
# endregion

# region Transformer
encoder_kwargs = dict(num_layers=1, num_attention_heads=8, intermediate_size=32,
    activation='relu', dropout_rate=0.0, attention_dropout_rate=0.0,
    use_bias=False, norm_first=True, norm_epsilon=1e-06, intermediate_dropout=0.0)
def create_transformer():
    encoder = tfm.nlp.models.TransformerEncoder(
        **encoder_kwargs)

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.GlobalAveragePooling1D(),
        create_predictormlp()
        ])
    return model
#endregion

#region CoRelNet
def create_corelnet():
    corelnet = tf.keras.layers.Lambda(lambda x: tf.matmul(x, x, transpose_b=True), name='similarity_matrix')

    model = tf.keras.Sequential(
        [
            corelnet,
            tf.keras.layers.Softmax(axis=-1, name='softmax'),
            tf.keras.layers.Flatten(name='flatten'),
            create_predictormlp()
            ],
        name='corelnet')
    return model

def create_nosoftmax_corelnet():
    corelnet = tf.keras.layers.Lambda(lambda x: tf.matmul(x, x, transpose_b=True), name='similarity_matrix')

    model = tf.keras.Sequential(
        [
            corelnet,
            # tf.keras.layers.Softmax(axis=-1, name='softmax'),
            tf.keras.layers.Flatten(name='flatten'),
            create_predictormlp()
            ],
        name='corelnet')
    return model
#endregion

#region PrediNet
predinet_kwargs = dict(key_dim=4, n_heads=4, n_relations=16, add_temp_tag=False)
def create_predinet():
    model = tf.keras.Sequential(
        [
            PrediNet(**predinet_kwargs),
            # tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1'),
            # tf.keras.layers.Dense(2, name='output')
            create_predictormlp()
            ],
        name='predinet')

    return model
#endregion

# region Abstractor
abstractor_kwargs = dict(num_layers=1, num_heads=8, dff=64, use_self_attn=False, dropout_rate=0.)
def create_abstractor():
    abstractor = RelationalAbstracter(**abstractor_kwargs)

    model = tf.keras.Sequential([
        abstractor,
        tf.keras.layers.GlobalAveragePooling1D(),
        create_predictormlp()
        ])
    return model
# endregion

# region LSTM
lstm_kwargs = dict(units=128, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)
def create_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(**lstm_kwargs),
        create_predictormlp()
        ])
    return model
# endregion

# region LSTM
gru_kwargs = dict(units=128, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)
def create_gru():
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(**gru_kwargs),
        create_predictormlp()
        ])
    return model
# endregion

# region GCN
gcn_kwargs = dict(channels=128, n_layers=2, dense_dim=128)
def create_gcn():
    from spektral.layers import GCNConv
    class GCNModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.convs = [GCNConv(gcn_kwargs['channels']) for _ in range(gcn_kwargs['n_layers'])]
            self.denses = [tf.keras.layers.Dense(gcn_kwargs['dense_dim'], activation='relu') for _ in range(gcn_kwargs['n_layers'])]
            self.pool = tf.keras.layers.GlobalAveragePooling1D()
            self.mlp_predictor = create_predictormlp()

        def call(self, inputs):
            x, a = inputs
            for conv, dense in zip(self.convs, self.denses):
                x = conv([x, a])
                x = dense(x)
            x = self.pool(x)
            out = self.mlp_predictor(x)
            return out

    return GCNModel()

# endregion


# put all model creators into a dictionary to interface with `eval_learning_curve.py`
model_creators = dict(
    relconvnet=create_relconvnet,
    relconvnet_maxpooling=create_relconvnet_maxpooling,
    transformer=create_transformer,
    corelnet=create_corelnet,
    nosoftmax_corelnet=create_nosoftmax_corelnet,
    predinet=create_predinet,
    lstm=create_lstm,
    gru=create_gru,
    gcn=create_gcn,
    abstractor=create_abstractor
    )