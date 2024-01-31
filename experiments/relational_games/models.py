import itertools
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
from cnn_embedder import CNNEmbedder

import sys; sys.path.append('..'); sys.path.append('../..')
from relational_neural_networks.mdipr import MultiDimInnerProdRelation
from relational_neural_networks.relational_graphlet_convolution import RelationalGraphletConvolution, RelationalGraphletConvolutionGroupAttn
from relational_neural_networks.grouping_layers import TemporalGrouping, FeatureGrouping
from relational_neural_networks.tcn import TCN, GroupTCN
from relational_neural_networks.predinet import PrediNet
from misc.abstractor import RelationalAbstracter

# global parameters
cnn_embedder_kwargs = dict(n_f=(16,16), s_f=(3,3), pool_size=2)
hidden_dense_size = 64

# RelConvNet

def create_relconvnet(normalizer=None, freeze_embedder=False, object_selection=None):
    relconv_mhr_kwargs = dict(rel_dim=16, proj_dim=4, symmetric=True)
    relconv_kwargs = dict(n_filters=16, graphlet_size=3,
            symmetric_inner_prod=False)
    object_selector = get_obj_selector(object_selection)
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    mhr1 = MultiDimInnerProdRelation(**relconv_mhr_kwargs, name='mhr1')
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

def create_relconvnet_groupattn():
    relconv_kwargs = dict(
        n_filters=16, graphlet_size=3, n_groups=8,
        mdipr_kwargs=dict(rel_dim=1, proj_dim=16, symmetric=True), # NOTE: changed proj_dim from 4
        group_attn_key_dim=32, group_attn_key='pos', symmetric_inner_prod=False, permutation_aggregator='max',
        filter_initializer='random_normal', entropy_reg=True, entropy_reg_scale=0.05)
    mdipr2_kwargs = dict(rel_dim=8, proj_dim=16, symmetric=True)

    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)

    rel_conv1 = RelationalGraphletConvolutionGroupAttn(
        **relconv_kwargs, name='rgc')

    mdipr2 = MultiDimInnerProdRelation(**mdipr2_kwargs, name='mdipr2')
    # rel_conv2 = RelationalGraphletConvolutionGroupAttn(
    #     **relconv_kwargs, name='rgc')

    model = tf.keras.Sequential([
        cnn_embedder,
        rel_conv1,
        mdipr2,
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu', name='hidden_dense1'), # NOTE: changed from hidden_dense_size = 64
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, name='output')
        ], name='relconv'
    )

    return model

def create_tempgroup_relconvnet(normalizer=None, freeze_embedder=False, object_selection=None):
    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.object_selector = get_obj_selector(object_selection) # NOTE: this is ignored for this model
            self.normalizer = get_normalizer(normalizer)
            self.cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
            self.cnn_embedder.trainable = not freeze_embedder
            self.mhr = MultiDimInnerProdRelation(**relconv_mhr_kwargs, name='mhr1')
            self.relconv = RelationalGraphletConvolution(**relconv_kwargs,
                groups='combinations', beta=1/9, group_normalizer='sparsemax', name='rgc1')
            self.grouper = TemporalGrouping(num_groups=16, weight_initializer='glorot_uniform', name='grouper')
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(hidden_dense_size, activation='relu')
            self.dense2 = tf.keras.layers.Dense(2, activation=None)

        def call(self, inputs):
            objs = self.object_selector(inputs)
            objs = self.cnn_embedder(objs)
            objs = self.normalizer(objs)
            reltensor = self.mhr(objs)
            groups = self.grouper(objs)
            conv = self.relconv(reltensor, groups=groups)
            x = self.flatten(conv)
            x = self.dense1(x)
            x = self.dense2(x)

            return x

    model = Model()

    return model

def create_featuregroup_relconvnet(normalizer=None, freeze_embedder=False, object_selection=None):
    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.object_selector = get_obj_selector(object_selection) # NOTE: this is ignored for this model
            self.normalizer = get_normalizer(normalizer)
            self.cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
            self.cnn_embedder.trainable = not freeze_embedder
            self.mhr = MultiDimInnerProdRelation(**relconv_mhr_kwargs, name='mhr1')
            self.relconv = RelationalGraphletConvolution(**relconv_kwargs,
                groups='combinations', beta=1/9, group_normalizer='sparsemax', name='rgc1')
            self.grouper = FeatureGrouping(num_groups=16, mlp_shape=(32,32),
                mlp_activations='relu', use_pos=True, name='grouper')
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(hidden_dense_size, activation='relu')
            self.dense2 = tf.keras.layers.Dense(2, activation=None)

        def call(self, inputs):
            objs = self.object_selector(inputs)
            objs = self.cnn_embedder(objs)
            objs = self.normalizer(objs)
            reltensor = self.mhr(objs)
            groups = self.grouper(objs)
            conv = self.relconv(reltensor, groups=groups)
            x = self.flatten(conv)
            x = self.dense1(x)
            x = self.dense2(x)

            return x

    model = Model()

    return model

def create_contextgroup_relconvnet(normalizer=None, freeze_embedder=False, object_selection=None):
    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # embedder
            self.object_selector = get_obj_selector(object_selection) # NOTE: this is ignored for this model
            self.normalizer = get_normalizer(normalizer)
            self.cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
            self.cnn_embedder.trainable = not freeze_embedder

            self.mhr = MultiDimInnerProdRelation(**relconv_mhr_kwargs, name='mhr1')
            self.relconv = RelationalGraphletConvolution(**relconv_kwargs,
                groups='combinations', beta=1/9, group_normalizer='sparsemax', name='rgc1')

            # grouping layers
            self.selfattn = tfm.nlp.layers.TransformerEncoderBlock(
                        num_attention_heads=4, inner_dim=32, inner_activation='relu')
            self.grouper = FeatureGrouping(num_groups=16, mlp_shape=(32,32),
                mlp_activations='relu', use_pos=True, name='grouper')

            # output MLP
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(hidden_dense_size, activation='relu')
            self.dense2 = tf.keras.layers.Dense(2, activation=None)


        def call(self, inputs):
            objs = self.object_selector(inputs)
            objs = self.cnn_embedder(objs)
            objs = self.normalizer(objs)
            reltensor = self.mhr(objs)
            groups = self.grouper(self.selfattn(objs))
            conv = self.relconv(reltensor, groups=groups)
            x = self.flatten(conv)
            x = self.dense1(x)
            x = self.dense2(x)

            return x

    model = Model()

    return model

def create_randomgroup_relconvnet(normalizer=None, freeze_embedder=False, object_selection=None):
    object_selector = get_obj_selector(object_selection) # NOTE: this is ignored for this model
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    mhr1 = MultiDimInnerProdRelation(**relconv_mhr_kwargs, name='mhr1')

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
def create_transformer(normalizer=None, freeze_embedder=False, object_selection=None):
    encoder_kwargs = dict(num_layers=1, num_attention_heads=8, intermediate_size=32,
        activation='relu', dropout_rate=0.0, attention_dropout_rate=0.0,
        use_bias=False, norm_first=True, norm_epsilon=1e-06, intermediate_dropout=0.0)
    object_selector = get_obj_selector(object_selection)
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    encoder = tfm.nlp.models.TransformerEncoder(
        **encoder_kwargs)

    model = tf.keras.Sequential([
        object_selector,
        cnn_embedder,
        normalizer,
        encoder,
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2)])
    return model

### Abstractor
def create_abstractor(normalizer=None, freeze_embedder=False, object_selection=None):
    abstractor_kwargs = dict(num_layers=1, num_heads=8, dff=64, use_self_attn=False, dropout_rate=0.)
    object_selector = get_obj_selector(object_selection)
    normalizer = get_normalizer(normalizer)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    cnn_embedder.trainable = not freeze_embedder

    abstractor = RelationalAbstracter(**abstractor_kwargs)

    model = tf.keras.Sequential([
        object_selector,
        cnn_embedder,
        normalizer,
        abstractor,
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2)
        ])
    return model

# NOTE: for the models below, normalizer, freeze_embedder, object_selection are ignored...
# those were used initially for experimentation with models above,
#  but were later removed from all experiments reported in the paper

# region LSTM
def create_lstm(normalizer=None, freeze_embedder=False, object_selection=None):
    lstm_kwargs = dict(units=32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    model = tf.keras.Sequential([
        cnn_embedder,
        tf.keras.layers.LSTM(**lstm_kwargs),
        tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1'),
        tf.keras.layers.Dense(2, name='output')
        ])
    return model
# endregion

# region LSTM
def create_gru(normalizer=None, freeze_embedder=False, object_selection=None):
    gru_kwargs = dict(units=32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)
    cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
    model = tf.keras.Sequential([
        cnn_embedder,
        tf.keras.layers.GRU(**gru_kwargs),
        tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1'),
        tf.keras.layers.Dense(2, name='output')
        ])
    return model
# endregion

# region GCN
def create_gcn(normalizer=None, freeze_embedder=False, object_selection=None):
    gcn_kwargs = dict(channels=32, n_layers=2, dense_dim=32)
    from spektral.layers import GCNConv
    class GCNModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
            self.convs = [GCNConv(gcn_kwargs['channels']) for _ in range(gcn_kwargs['n_layers'])]
            self.denses = [tf.keras.layers.Dense(gcn_kwargs['dense_dim'], activation='relu') for _ in range(gcn_kwargs['n_layers'])]
            self.pool = tf.keras.layers.GlobalAveragePooling1D()
            self.hidden_dense1 = tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1')
            self.out = tf.keras.layers.Dense(2, name='output')

        def build(self, input_shape):
            _, self.n_objs, *_ = input_shape

        def call(self, inputs):
            x = self.cnn_embedder(inputs)
            a = tf.ones(shape=(tf.shape(x)[0], self.n_objs, self.n_objs))
            for conv, dense in zip(self.convs, self.denses):
                x = conv([x, a])
                x = dense(x)
            x = self.pool(x)
            x = self.hidden_dense1(x)
            out = self.out(x)
            return out

    return GCNModel()
# endregion

# region GAT
def create_gat(normalizer=None, freeze_embedder=False, object_selection=None):
    gat_kwargs = dict(channels=32, n_layers=2, dense_dim=32)
    from spektral.layers import GATConv
    class GATModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
            self.convs = [GATConv(gat_kwargs['channels']) for _ in range(gat_kwargs['n_layers'])]
            self.denses = [tf.keras.layers.Dense(gat_kwargs['dense_dim'], activation='relu') for _ in range(gat_kwargs['n_layers'])]
            self.pool = tf.keras.layers.GlobalAveragePooling1D()
            self.hidden_dense1 = tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1')
            self.out = tf.keras.layers.Dense(2, name='output')

        def build(self, input_shape):
            _, self.n_objs, *_ = input_shape

        def call(self, inputs):
            x = self.cnn_embedder(inputs)
            a = tf.ones(shape=(tf.shape(x)[0], self.n_objs, self.n_objs))
            for conv, dense in zip(self.convs, self.denses):
                x = conv([x, a])
                x = dense(x)
            x = self.pool(x)
            x = self.hidden_dense1(x)
            out = self.out(x)
            return out

    return GATModel()
# endregion GAT

# region  GIN
def create_gin(normalizer=None, freeze_embedder=False, object_selection=None):
    gin_kwargs = dict(channels=32, n_layers=2, dense_dim=32)
    from spektral.layers import GINConvBatch
    class GINModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.cnn_embedder = CNNEmbedder(**cnn_embedder_kwargs)
            self.convs = [GINConvBatch(gin_kwargs['channels']) for _ in range(gin_kwargs['n_layers'])]
            self.denses = [tf.keras.layers.Dense(gin_kwargs['dense_dim'], activation='relu') for _ in range(gin_kwargs['n_layers'])]
            self.pool = tf.keras.layers.GlobalAveragePooling1D()
            self.hidden_dense1 = tf.keras.layers.Dense(hidden_dense_size, activation='relu', name='hidden_dense1')
            self.out = tf.keras.layers.Dense(2, name='output')

        def build(self, input_shape):
            _, self.n_objs, *_ = input_shape

        def call(self, inputs):
            x = self.cnn_embedder(inputs)
            a = tf.ones(shape=(tf.shape(x)[0], self.n_objs, self.n_objs))
            for conv, dense in zip(self.convs, self.denses):
                x = conv([x, a])
                x = dense(x)
            x = self.pool(x)
            x = self.hidden_dense1(x)
            out = self.out(x)
            return out

    return GINModel()
# endregion GIN



# region utilities for model configurations
def get_normalizer(normalizer):
    if normalizer is None or normalizer=='None':
        return tf.keras.layers.Identity(name='identity_normalizer')
    elif normalizer == 'l2':
        return tf.keras.layers.UnitNormalization(name='l2_norm')
    elif normalizer == 'tcn':
        return TCN(name='tcn')
    else:
        raise ValueError(f'unknown normalizer {normalizer}')

def get_obj_selector(object_selection):
    if object_selection is None or object_selection=='None':
        return tf.keras.layers.Identity(name='identity_obj_selector')
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
    if normalizer is not None or object_selection=='None':
        group_name += f'-{normalizer}'
    if freeze_embedder:
        group_name += '-freeze_embedder'
    if object_selection is not None:
        group_name += '-w_obj_selection'
    return group_name
#endregion

# put all model creators into a dictionary to interface with `eval_learning_curve.py`
model_creators = dict(
    relconvnet=create_relconvnet,
    tempgroup_relconvnet=create_tempgroup_relconvnet,
    featuregroup_relconvnet=create_featuregroup_relconvnet,
    contextgroup_relconvnet=create_contextgroup_relconvnet,
    randomgroup_relconvnet=create_randomgroup_relconvnet,
    transformer=create_transformer,
    corelnet=create_corelnet,
    nosoftmax_corelnet=create_nosoftmaxcorelnet,
    predinet=create_predinet,
    abstractor=create_abstractor,
    lstm=create_lstm,
    gru=create_gru,
    gcn=create_gcn,
    gat=create_gat,
    gin=create_gin,
    relconvnet_groupattn=create_relconvnet_groupattn
    )