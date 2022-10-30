import tensorflow as tf

import numpy as np

import itertools


class RelationalLayer(tf.keras.layers.Layer):
    """
    RelationalLayer

    Computes the relation tensor for a sequence of entities
    across attributes given by a set of encoders.
    """

    def __init__(self, attribute_encoder_constructors, name=None):
        """
        create RelationalLayer

        Parameters
        ----------
        attribute_encoder_constructors : List[callable]
            list of constructors for encoder layers
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)
        self.attr_enc_constructors = attribute_encoder_constructors

        # construct encoders
        self.entity_encoders = [
            EntityEncoder(attr_enc_constructor, name=f'attr_{i}_entity_encoder')
            for i, attr_enc_constructor in enumerate(self.attr_enc_constructors)
            ]

        # relational inner product layer
        self.rel_inner_prod = RelInnerProduct(name='relational_inner_product')


    def call(self, inputs):

        relation_matrices = []

        # compute relation matrices for each attribute encoder
        for entity_encoder in self.entity_encoders:
            # transform entities' representation using encoder
            attr_zs = entity_encoder(inputs)
            # compute the relation matrix and append to list
            attr_R = self.rel_inner_prod(attr_zs)
            relation_matrices.append(attr_R)

        # construct relation tensor from relation matrices
        relation_tensor = tf.stack(relation_matrices, axis=-1)

        return relation_tensor


class RelInnerProduct(tf.keras.layers.Layer):
    """
    Relational Inner Product Layer.

    computes the pairwise inner product similarities between the feature vectors
    of a sequence of entities.
    """

    def __init__(self, name=None):
        """
        create RelInnerProduct Layer.

        Parameters
        ----------
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)

    def call(self, inputs):
        return tf.matmul(inputs, tf.transpose(inputs, perm=(0,2,1)))


class EntityEncoder(tf.keras.layers.Layer):
    """
    Entity Encoder Layer

    transforms the feature vector representation of each entity in a sequence
    using a given encoder.
    """

    def __init__(self, encoder_constructor, name=None):
        """
        create EntityEncoder layer

        Parameters
        ----------
        encoder_constructor : callable
            function which creates the encoder layer
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)
        self.encoder = encoder_constructor()

    def call(self, inputs):
        return tf.map_fn(self.encoder, inputs)



class LinearProjectionEncoder(tf.keras.layers.Layer):
    """
    Linear Projection Encoder.

    Learns a 1-dimensional linear subspace of feaure space and projects along it.
    """

    def __init__(self, name=None):
        """
        create LinearProjectionEncoder layer.

        Parameters
        ----------
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)

    def build(self, input_shape):

        w_init = tf.random_normal_initializer() # projection vector's intializer
        self.in_dim = input_shape[-1] # dimension of features

        # define learnable projection vector
        self.proj_vec = tf.Variable(
            initial_value=w_init(shape=(self.in_dim,)),
            trainable=True, name='proj_vec')

    def call(self, inputs):

        # normalized projection vector
        norm_proj_vec, _ = tf.linalg.normalize(self.proj_vec, ord='euclidean')

        #project inputs along normalized proj_vec
        projection_length = tf.tensordot(tf.transpose(norm_proj_vec), inputs, axes=(0,1))
        projected_inputs = tf.expand_dims(projection_length, axis=-1) * norm_proj_vec

        return projected_inputs


class MLPEncoder(tf.keras.layers.Layer):
    """
    MLP Encoder Layer.

    Transforms input feature vector via a series of dense feedforward neural network layers.
    """

    def __init__(self, layer_sizes, activation='relu', name=None):
        """
        create MLP encoder layer.

        Parameters
        ----------
        layer_sizes : List[int]
            list of the number of units at each dense layer.
        activation : (list of) tensorflow activation function, optional
            tensorflow activation function or str name. if list matching
            `layer_sizes` is given, a different activation is used at each layer. by default 'relu'
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)

        self.layer_sizes = layer_sizes
        self.activation = activation

        # check that `activation`` is valid
        if isinstance(activation, list):
            if len(activation) != len(layer_sizes):
                raise ValueError(
                    '`activation` must either be a single activation paramter '
                    'or a list matching `layer_sizes`')
        else:
            # activation for each layer
            self.activation = [self.activation] * len(self.layer_sizes)

        # construct series of dense layers
        self.layers = []
        for layer_size, activation in zip(self.layer_sizes, self.activation):
            self.layers.append(tf.keras.layers.Dense(layer_size, activation=activation))


    def call(self, inputs):
        x = inputs

        # iteratively transform input via dense NN layers
        for layer in self.layers:
            x = layer(x)

        return x

# TODO: test this. make sure its doing what it's supposed to.
# can implementation be made more efficient? (single call to tensordot without stacking possible?)
# TODO: consider adding option to exclude (i, i) from grouped relation vector
# TODO: add different options for normalization in computing alpha_k from group_logits?
# (e.g.: softmax, no normalization, divide by L1 norm, etc.)
class GroupLayer(tf.keras.layers.Layer):
    """
    Grouping layer in relational neural network framework.
    """

    def __init__(self, num_groups, name=None):
        """
        create GroupLayer.

        Given a relation tensor as input, GroupLayer groups entities and
        produces feature vectors for each group which describes the relations within it.

        Parameters
        ----------
        num_groups : int
            number of groups.
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)
        self.num_groups = num_groups

    def build(self, input_shape):

        batch_size, self.n_entities, _, self.rel_dim = input_shape

        w_init = tf.random_normal_initializer() # group logits intializer

        # define learnable group logits
        self.group_logits = [
            tf.Variable(
                initial_value=w_init(shape=(self.n_entities,1)),
                trainable=True,
                name=f'group_{k}_logit')
            for k in range(self.num_groups)
            ]

        self.out_shape = (batch_size, self.num_groups, self.rel_dim)


    def call(self, inputs):

        zs = []
        for k in range(self.num_groups):
            # normalized group membership logits
            alpha_k = tf.nn.softmax(self.group_logits[k], axis=0)

            z_k = sum(
                alpha_k[i]*alpha_k[j]*inputs[:, i, j] for i, j in
                itertools.product(range(self.n_entities), repeat=2))

            ## alternative implementation below
            ## (prelim tests show the two methods are about the same speed)
            # grouped_rels = [
            #   tf.matmul(tf.matmul(tf.transpose(alpha_k), inputs[:, :, :, i]), alpha_k)
            #   for i in range(self.rel_dim)]
            # grouped_rels = tf.stack(grouped_rels, axis=-1)
            # z_k = tf.squeeze(grouped_rels)

            zs.append(z_k)


        zs = tf.stack(zs, axis=1)

        zs.set_shape(self.out_shape)


        return zs

class FlattenTriangular(tf.keras.layers.Layer):
    """
    Triangular Flatten Layer.
    """
    def __init__(self, include_diag=True, name=None):
        """
        Create FlattenTriangular layer.

        Extracts triangle from [None, n_e, n_e, d_r] tensor as a flattened vector.
        Useful for tensors which are symmetric in the first two axes.

        Parameters
        ----------
        include_diag : bool, optional
            Whether to include the diagonal in flattened vector, by default True
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        self.include_diag = include_diag

    def build(self, input_shape):
        batch_dim, n_e, _, d_r = input_shape

        if n_e != _:
            raise ValueError('Input is not a square relation tensor')

        # create mask for extracting triangle
        if self.include_diag:
            self.mask = tf.convert_to_tensor(np.tril(np.ones(shape=(n_e,n_e)), k=0), dtype=bool)
        else:
            self.mask = tf.convert_to_tensor(np.tril(np.ones(shape=(n_e, n_e)), k=-1), dtype=bool)

        self.out_dim = np.sum(self.mask.numpy()) * d_r
        self.out_shape = (batch_dim, self.out_dim)


    def call(self, inputs):
        triangle = tf.boolean_mask(inputs, self.mask, axis=1) # get triangle for each d_r dim
        flattened_vec = self.flatten(triangle) # flatten d_r vecs into single
        flattened_vec.set_shape(self.out_shape) # set shape so tensorflow can do its thing

        return flattened_vec