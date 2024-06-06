"""
An implementation of the Relation Network layer as described in the paper
"A simple neural network module for relational reasoning" by Santoro et al. (2017).
"""

import tensorflow as tf
from tensorflow.keras import layers
import itertools
import numpy as np

class RelationNetwork(tf.keras.layers.Layer):
    def __init__(self, neurons, activation, **kwargs):
        super(RelationNetwork, self).__init__(**kwargs)
        self.neurons = neurons
        self.activation = activation

    def build(self, input_shape):
        batch_dim, self.n_objects, self.object_dim = input_shape

        self.object_pairs_idx = tf.convert_to_tensor(
            list(itertools.product(range(self.n_objects), repeat=2))
        )

        self.relation_mlp = tf.keras.Sequential(
            [layers.Dense(n, activation=self.activation) for n in self.neurons]
        )

        self.concat_pair = tf.keras.layers.Reshape(target_shape=(self.n_objects**2, 2*self.object_dim))

    def call(self, inputs):
        object_pairs = tf.gather(inputs, self.object_pairs_idx, axis=1)
        object_pairs_concat = self.concat_pair(object_pairs)
        object_pair_rels = self.relation_mlp(object_pairs_concat)
        agg_rels = tf.reduce_sum(object_pair_rels, axis=1)
        return agg_rels
