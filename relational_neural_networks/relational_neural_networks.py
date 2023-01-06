import tensorflow as tf

import numpy as np

import itertools
import copy



class RelationalLayer(tf.keras.layers.Layer):
    """
    RelationalLayer

    Computes the relation tensor for a sequence of entities
    across attributes given by a set of encoders.
    """

    def __init__(self, encoder_layers, name=None):
        """
        create RelationalLayer

        Parameters
        ----------
        encoder_layers : List[tf.keras.layers.Layer]
            list of encoder layers
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)

        self.encoder_layers = encoder_layers

        # construct encoders
        self.entity_encoders = [
            tf.keras.layers.TimeDistributed(attr_enc, name=f'attr_{i}_entity_encoder')
            for i, attr_enc in enumerate(encoder_layers)
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

    def get_config(self):
        config = {"encoder_layers": [tf.keras.layers.serialize(encoder_layer) for encoder_layer in self.encoder_layers]}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):

        # Avoid mutating the input dict
        config = copy.deepcopy(config)
        encoder_layers = [tf.keras.layers.deserialize(encoder_layer, custom_objects=custom_objects)
                          for encoder_layer in config.pop("encoder_layers")]
        return cls(encoder_layers, **config)


class RelInnerProduct(tf.keras.layers.Layer):
    """
    Relational Inner Product Layer.

    Computes the pairwise inner product similarities between the feature vectors
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

    def get_config(self):
        config = super(GroupLayer, self).get_config()
        config.update({'num_groups': self.num_groups})

        return config


class FlattenTriangular(tf.keras.layers.Layer):
    """
    Triangular Flattening Layer.
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

    def get_config(self):
        config = super(FlattenTriangular, self).get_config()
        config.update({'include_diag': self.include_diag})

        return config
